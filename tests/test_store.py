"""Feature store semantics on hand-written logs: every feature means what its name says, at the right time."""

from __future__ import annotations

import math
import pickle

import numpy as np
import pytest

from fraudnet.features.replay import replay
from fraudnet.features.schema import FEATURE_INDEX, FEATURE_NAMES, GROUPS, N_FEATURES, columns_for
from fraudnet.features.store import LabelEvent, OnlineFeatureStore
from fraudnet.sim.config import DAY, HOUR, MINUTE
from tests.helpers import make_dataset

T0 = 10 * DAY + 12 * HOUR


def _features(rows: list[dict]) -> np.ndarray:
    return replay(make_dataset(rows)).table.X


def _f(X: np.ndarray, i: int, name: str) -> float:
    return float(X[i, FEATURE_INDEX[name]])


def test_schema_names_are_unique_and_grouped():
    assert len(set(FEATURE_NAMES)) == N_FEATURES == 53
    assert sorted(columns_for([g.name for g in GROUPS])) == list(range(N_FEATURES))
    with pytest.raises(KeyError):
        columns_for(["no_such_group"])


def test_velocity_counts_only_earlier_transactions():
    X = _features([{"ts": T0 + k * 10 * MINUTE, "account": 1, "device": 1} for k in range(4)])
    assert [_f(X, i, "acct_cnt_1h") for i in range(4)] == [0.0, 1.0, 2.0, 3.0]
    assert _f(X, 0, "acct_txn_count") == 0.0
    assert _f(X, 3, "acct_txn_count") == pytest.approx(math.log1p(3))


def test_windows_expire():
    X = _features(
        [
            {"ts": T0, "account": 1, "device": 1},
            {"ts": T0 + 2 * HOUR, "account": 1, "device": 1},
            {"ts": T0 + 2 * DAY, "account": 1, "device": 1},
        ]
    )
    assert _f(X, 1, "acct_cnt_1h") == 0.0 and _f(X, 1, "acct_cnt_24h") == 1.0
    assert _f(X, 2, "acct_cnt_24h") == 0.0 and _f(X, 2, "acct_cnt_7d") == 2.0


def test_new_device_flag_and_shared_device_counts():
    X = _features([
        {"ts": T0, "account": 1, "device": 7},
        {"ts": T0 + 60, "account": 1, "device": 7},
        {"ts": T0 + 120, "account": 2, "device": 7},
        {"ts": T0 + 180, "account": 3, "device": 7},
    ])  # fmt: skip
    assert [_f(X, i, "device_new_for_acct") for i in range(4)] == [1.0, 0.0, 1.0, 1.0]
    assert [_f(X, i, "device_accts_total") for i in range(4)] == [0.0, 1.0, 1.0, 2.0]
    assert _f(X, 3, "device_accts_24h") == 2.0
    assert _f(X, 3, "device_cnt_1h") == 3.0


def test_device_neighbours_reach_accounts_through_shared_devices():
    X = _features([
        {"ts": T0, "account": 2, "device": 7},
        {"ts": T0 + 60, "account": 3, "device": 7},
        {"ts": T0 + 120, "account": 1, "device": 7},
        {"ts": T0 + 180, "account": 1, "device": 8},
    ])  # fmt: skip
    assert _f(X, 3, "acct_devices_total") == 1.0
    assert _f(X, 3, "acct_device_neighbours") == pytest.approx(math.log1p(2))


def test_pass_through_ratio_and_new_payee():
    X = _features([
        {"ts": T0, "account": 1, "device": 1, "dst": 2, "amount": 1000.0},
        {"ts": T0 + 30 * MINUTE, "account": 2, "device": 2, "dst": 3, "amount": 900.0},
        {"ts": T0 + 40 * MINUTE, "account": 2, "device": 2, "dst": 3, "amount": 50.0},
    ])  # fmt: skip
    assert _f(X, 1, "passthrough_ratio") == pytest.approx(0.9)
    assert _f(X, 1, "acct_inflow_24h") == pytest.approx(math.log1p(1000.0))
    assert _f(X, 1, "log_secs_since_inflow") == pytest.approx(math.log1p(30 * MINUTE))
    assert _f(X, 1, "payee_new") == 1.0 and _f(X, 2, "payee_new") == 0.0
    assert _f(X, 2, "acct_outflow_24h") == pytest.approx(math.log1p(900.0))
    assert _f(X, 1, "dst_senders_7d") == 0.0 and _f(X, 0, "dst_age_days") > 0


def test_payee_senders_count_distinct_accounts():
    X = _features(
        [{"ts": T0 + k * 60, "account": k + 1, "device": k + 1, "dst": 9, "amount": 100.0} for k in range(4)]
    )
    assert [_f(X, i, "dst_senders_7d") for i in range(4)] == [0.0, 1.0, 2.0, 3.0]


def test_small_charges_at_a_merchant_are_counted_per_hour():
    rows = [
        {"ts": T0 + k * 30, "account": k + 1, "device": 50, "merchant": 4, "amount": 1.0} for k in range(6)
    ]
    rows.append({"ts": T0 + 2 * HOUR, "account": 9, "device": 9, "merchant": 4, "amount": 1.0})
    X = _features(rows)
    assert _f(X, 5, "merchant_small_cnt_1h") == 5.0
    assert _f(X, 5, "merchant_new_device_share_1h") == 1.0
    assert _f(X, 6, "merchant_small_cnt_1h") == 0.0


def test_amount_z_compares_with_the_accounts_own_history():
    rows = [
        {"ts": T0 + k * HOUR, "account": 1, "device": 1, "amount": a}
        for k, a in enumerate([20.0, 22.0, 18.0, 21.0, 400.0])
    ]
    X = _features(rows)
    assert _f(X, 2, "amount_z") == 0.0, "needs three earlier transactions"
    assert _f(X, 4, "amount_z") > 5.0
    assert _f(X, 4, "amount_vs_max") > 2.0


def test_reports_are_visible_only_after_their_report_time():
    report = T0 + 2 * HOUR
    X = _features([
        {"ts": T0, "account": 1, "device": 5, "pattern": 3, "report_ts": report},
        {"ts": T0 + HOUR, "account": 2, "device": 5},
        {"ts": report, "account": 3, "device": 5},
        {"ts": report + 1, "account": 4, "device": 5},
    ])  # fmt: skip
    assert _f(X, 1, "device_known_fraud") == 0.0, "report not arrived yet"
    assert _f(X, 2, "device_known_fraud") == 0.0, "a report at the same second is not yet visible"
    assert _f(X, 3, "device_known_fraud") == pytest.approx(math.log1p(1))


def test_instant_schedule_leaks_unreported_labels():
    rows = [
        {"ts": T0, "account": 1, "device": 5, "pattern": 3, "report_ts": T0 + 10 * DAY},
        {"ts": T0 + HOUR, "account": 2, "device": 5},
    ]
    honest = replay(make_dataset(rows)).table.X
    leaky = replay(make_dataset(rows), schedule="instant").table.X
    assert _f(honest, 1, "device_known_fraud") == 0.0
    assert _f(leaky, 1, "device_known_fraud") > 0.0


def test_store_rejects_events_older_than_its_clock():
    store = OnlineFeatureStore(np.zeros(5, dtype=np.int64), np.zeros(3, dtype=np.int8))
    store.process(T0, 1, 1, 0, -1, 10.0, 1)
    with pytest.raises(ValueError, match="older"):
        store.advance_to(T0 - 1)


def test_unknown_account_is_new_at_first_sight():
    store = OnlineFeatureStore(np.zeros(5, dtype=np.int64), np.zeros(3, dtype=np.int8))
    row = store.process(T0, 999_999, 1, 0, -1, 10.0, 1)
    assert row[FEATURE_INDEX["acct_age_days"]] == 0.0
    assert store.signup_of(999_999, T0 + DAY) == T0


def test_feature_rows_are_complete_and_finite(first_month_features):
    assert first_month_features.shape[1] == N_FEATURES
    assert np.isfinite(first_month_features).all()


def test_snapshots_refresh_at_day_boundaries_from_earlier_events(small_dataset):
    seen = []
    result = replay(small_dataset, stop_ts=5 * DAY)
    store = result.store
    store.on_snapshot = seen.append
    store.advance_to(6 * DAY + 1)
    assert [s.time for s in seen] == [5 * DAY, 6 * DAY]
    assert all(s.time % DAY == 0 for s in seen)
    window_ts = small_dataset.txns["ts"]
    assert seen[0].account_ids.size == np.unique(np.concatenate([
        small_dataset.txns["account"][window_ts < 5 * DAY],
        small_dataset.txns["dst"][(window_ts < 5 * DAY) & (small_dataset.txns["dst"] >= 0)],
    ])).size  # fmt: skip


def test_pickled_store_continues_identically(small_dataset):
    result = replay(small_dataset, stop_ts=12 * DAY)
    clone = pickle.loads(pickle.dumps(result.store))
    t = small_dataset.txns
    start = int(np.searchsorted(t["ts"], 12 * DAY))
    for i in range(start, start + 50):
        args = (
            int(t["ts"][i]),
            int(t["account"][i]),
            int(t["device"][i]),
            int(t["merchant"][i]),
            int(t["dst"][i]),
            float(t["amount"][i]),
            int(t["channel"][i]),
        )
        assert result.store.process(*args) == clone.process(*args)


def test_scheduled_report_updates_counters_after_clock_passes():
    store = OnlineFeatureStore(np.zeros(5, dtype=np.int64), np.zeros(3, dtype=np.int8))
    store.process(T0, 1, 7, 0, -1, 10.0, 1)
    store.schedule_label(LabelEvent(T0 + 10, 0, 1, 7, 0, -1))
    row_before = store.process(T0 + 10, 2, 7, 0, -1, 10.0, 1)
    row_after = store.process(T0 + 11, 3, 7, 0, -1, 10.0, 1)
    assert row_before[FEATURE_INDEX["device_known_fraud"]] == 0.0
    assert row_after[FEATURE_INDEX["device_known_fraud"]] > 0.0
    assert store.fraud_devices == {7} and store.labels_applied == 1
