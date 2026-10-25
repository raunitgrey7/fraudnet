"""The scoring API: contract, validation, auth, the alert loop, online and offline parity, concurrency."""

from __future__ import annotations

import threading
from dataclasses import replace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from fraudnet.config import Settings
from fraudnet.features.schema import FEATURE_NAMES
from fraudnet.serving.app import create_app
from fraudnet.serving.service import OutOfOrderError, Transaction, to_iso

PURCHASE = {"account_id": 12, "device_id": 12, "merchant_id": 3, "amount": 49.9, "channel": "ecom"}
TRANSFER = {"account_id": 12, "device_id": 12, "payee_account_id": 40, "amount": 900.0, "channel": "p2p"}


def _client(service, api_key: str | None = None) -> TestClient:
    settings = replace(Settings.from_env(), api_key=api_key, log_level="WARNING")
    return TestClient(create_app(settings, service))


@pytest.fixture
def client(make_service):
    with _client(make_service(feed=True)) as c:
        yield c


def test_health_and_models(client):
    health = client.get("/health").json()
    assert health["status"] == "ok" and health["models"] == ["rules", "gbdt", "graphsage"]
    models = client.get("/v1/models").json()
    assert models["default"] == "gbdt" and len(models["features"]) == len(FEATURE_NAMES)
    assert set(models["thresholds"]["gbdt"]) == {"review", "block"}


def test_score_purchase_returns_decision_reasons_and_request_id(client):
    response = client.post("/v1/score", json=PURCHASE, headers={"x-request-id": "abc-123"})
    assert response.status_code == 200
    body = response.json()
    assert body["decision"] in {"approve", "review", "block"}
    assert 0.0 <= body["score"] <= 1.0 and isinstance(body["reasons"], list)
    assert body["request_id"] == "abc-123" and response.headers["x-request-id"] == "abc-123"
    assert float(response.headers["x-process-time-ms"]) > 0


@pytest.mark.parametrize("model", ["rules", "gbdt", "graphsage"])
def test_every_model_scores_a_transfer(client, model):
    body = client.post("/v1/score", json={**TRANSFER, "model": model}).json()
    assert body["model"] == model and body["decision"] in {"approve", "review", "block"}


def test_features_are_returned_by_name_on_request(client):
    body = client.post("/v1/score", json={**PURCHASE, "include_features": True}).json()
    assert list(body["features"]) == list(FEATURE_NAMES)
    assert body["features"]["merchant_category"] >= 0


@pytest.mark.parametrize(
    ("payload", "fragment"),
    [
        ({**PURCHASE, "payee_account_id": 4}, "exactly one"),
        ({k: v for k, v in PURCHASE.items() if k != "merchant_id"}, "exactly one"),
        ({**PURCHASE, "channel": "p2p"}, "pos or ecom"),
        ({**TRANSFER, "channel": "ecom"}, "p2p"),
        ({**TRANSFER, "payee_account_id": 12}, "cannot pay itself"),
        ({**PURCHASE, "amount": -5}, "greater than 0"),
        ({**PURCHASE, "account_id": -1}, "account_id"),
        ({**PURCHASE, "surprise": 1}, "surprise"),
        ({**PURCHASE, "txn_id": "bad id with spaces"}, "txn_id"),
        ({**PURCHASE, "model": "xgboost"}, "model"),
    ],
)
def test_invalid_payloads_are_rejected_without_touching_the_store(client, payload, fragment):
    before = client.get("/v1/stats").json()["store"]["accounts"]
    response = client.post("/v1/score", json=payload)
    assert response.status_code == 422
    assert fragment in response.json()["detail"]
    assert client.get("/v1/stats").json()["store"]["accounts"] == before


@pytest.mark.parametrize("raw", ["1e999", "NaN", "Infinity"])
def test_non_finite_amounts_are_rejected(client, raw):
    body = '{"account_id": 1, "device_id": 1, "merchant_id": 3, "channel": "ecom", "amount": ' + raw + "}"
    response = client.post("/v1/score", content=body, headers={"content-type": "application/json"})
    assert response.status_code == 422
    assert "amount" in response.json()["detail"]


def test_stale_timestamp_is_a_conflict(client):
    response = client.post("/v1/score", json={**PURCHASE, "timestamp": "2026-07-02T00:00:00Z"})
    assert response.status_code == 409 and "older than the store clock" in response.json()["detail"]


def test_batch_scores_in_order_and_has_a_size_limit(client):
    items = [{**PURCHASE, "txn_id": f"b{i}"} for i in range(5)]
    body = client.post("/v1/score/batch", json={"transactions": items}).json()
    assert [r["txn_id"] for r in body["results"]] == [f"b{i}" for i in range(5)]
    too_many = {"transactions": [PURCHASE] * 501}
    assert client.post("/v1/score/batch", json=too_many).status_code == 422


def test_replay_fills_the_alert_queue_and_resolution_feeds_back(client):
    summary = client.post("/v1/replay", json={"n": 400}).json()
    assert summary["scored"] == 400 and summary["alerts"] >= 1
    queue = client.get("/v1/alerts").json()
    assert queue["count"] >= 1
    scores = [a["score"] for a in queue["alerts"]]
    assert scores == sorted(scores, reverse=True)
    alert = queue["alerts"][0]
    pending_before = client.get("/v1/stats").json()["store"]["labels_pending"]
    resolved = client.post(f"/v1/alerts/{alert['alert_id']}/resolve", json={"label": "fraud"})
    assert resolved.status_code == 200 and resolved.json()["status"] == "fraud"
    assert client.get("/v1/stats").json()["store"]["labels_pending"] == pending_before + 1
    again = client.post(f"/v1/alerts/{alert['alert_id']}/resolve", json={"label": "legit"})
    assert again.status_code == 404


def test_fraud_report_reaches_later_transactions_on_the_same_device(make_service):
    service = make_service()
    clock = service.store.clock
    first = service.score(Transaction(clock + 10, 3, 777_001, 2, -1, 25.0, 1, "x1"))
    assert service.report_fraud("x1")
    later = service.score(Transaction(clock + 20, 4, 777_001, 2, -1, 25.0, 1, "x2"))
    idx = FEATURE_NAMES.index("device_known_fraud")
    assert first.features[idx] == 0.0 and later.features[idx] > 0.0
    assert not service.report_fraud("missing")


def test_labels_endpoint(client):
    scored = client.post("/v1/score", json={**PURCHASE, "txn_id": "lab-1"}).json()
    assert client.post("/v1/labels", json={"txn_id": scored["txn_id"]}).status_code == 200
    assert client.post("/v1/labels", json={"txn_id": "never-seen"}).status_code == 404


def test_rings_and_network_views(client):
    client.post("/v1/replay", json={"n": 300})
    rings = client.get("/v1/rings").json()["rings"]
    for ring in rings:
        assert ring["accounts"] >= 3 and ring["devices"] >= 1
        assert {e["kind"] for e in ring["edges"]} <= {"device", "transfer"}
    network = client.get("/v1/accounts/12/network")
    assert network.status_code == 200
    assert any(n["id"] == "a12" for n in network.json()["nodes"])
    assert client.get("/v1/accounts/987654321/network").status_code == 404


def test_metrics_export_service_counters(client):
    client.post("/v1/score", json=PURCHASE)
    client.post("/v1/replay", json={"n": 20})
    text = client.get("/metrics").text
    scored = [line for line in text.splitlines() if line.startswith("fraudnet_scored_total{")]
    assert any('model="gbdt"' in line and 'decision="approve"' in line for line in scored)
    assert "fraudnet_http_requests_total" in text and "fraudnet_alerts_open" in text
    assert "fraudnet_score_latency_seconds_bucket" in text


def test_api_key_is_required_when_configured(make_service):
    with _client(make_service(), api_key="s3cret") as c:
        assert c.get("/health").status_code == 200
        assert c.get("/metrics").status_code == 200
        denied = c.post("/v1/score", json=PURCHASE)
        assert denied.status_code == 401 and denied.json()["detail"] == "invalid or missing API key"
        assert c.post("/v1/score", json=PURCHASE, headers={"x-api-key": "wrong"}).status_code == 401
        assert c.post("/v1/score", json=PURCHASE, headers={"x-api-key": "s3cret"}).status_code == 200


def test_dashboard_is_served(client):
    page = client.get("/")
    assert page.status_code == 200 and "fraudnet console" in page.text
    assert client.get("/static/app.js").status_code == 200


def test_online_features_equal_offline_replay(make_service, held_out, prepared):
    service = make_service()
    for i, txn in held_out(300):
        result = service.score(txn, "gbdt", explain=False)
        assert np.array_equal(result.features, prepared.table.X[i]), f"training/serving skew at row {i}"


def test_service_rejects_out_of_order_events(make_service, held_out):
    service = make_service()
    (_, first), (_, second) = held_out(2)
    service.score(second)
    if first.ts < second.ts:
        with pytest.raises(OutOfOrderError):
            service.score(first)


def test_concurrent_scoring_keeps_the_store_consistent(make_service):
    service = make_service()
    clock = service.store.clock
    errors: list[Exception] = []

    def worker(k: int) -> None:
        try:
            for j in range(40):
                service.score(Transaction(clock, 900_000 + k, 800_000 + k, 1, -1, 10.0 + j, 1, f"c{k}-{j}"))
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert all(service.store.accounts[900_000 + k].n == 40 for k in range(4))


def test_timestamps_round_trip_through_iso():
    assert to_iso(0) == "2026-07-01T00:00:00Z"
    assert to_iso(86_400 * 70) == "2026-09-09T00:00:00Z"
