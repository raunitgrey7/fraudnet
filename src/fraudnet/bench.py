"""Benchmark harness: simulate, replay, train, evaluate, ablate, prove point-in-time, time the API.

``python -m fraudnet.bench`` writes ``benchmarks/results.json`` (the source of truth) and
``benchmarks/results.md``, and with ``--update-readme`` copies the tables into README.md between markers.
``--check`` compares a results file with ``benchmarks/thresholds.json`` and exits non-zero on a breach, which
is the CI quality gate. ``--quick`` runs the same pipeline on the smaller CI world.

Every number is computed here from a seeded run; nothing is typed in by hand.
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import platform
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
import torch
from fastapi.testclient import TestClient
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits

from fraudnet import __version__
from fraudnet.config import ROOT, Settings
from fraudnet.evaluation.metrics import (
    CostModel,
    day_block_bootstrap,
    detection_metrics,
    paired_delta_bootstrap,
    recall_at_fpr,
    threshold_at_fpr,
    top_k_per_day,
)
from fraudnet.evaluation.report import check_thresholds, render_markdown, update_readme, update_report
from fraudnet.evaluation.split import TimeSplit
from fraudnet.features.leakage import check_point_in_time, known_at
from fraudnet.features.replay import GnnSnapshot, replay
from fraudnet.features.schema import GRAPH_GROUPS, N_FEATURES, TABULAR_GROUPS, columns_for
from fraudnet.logging_utils import configure_logging
from fraudnet.models.gbdt import GBDTModel
from fraudnet.models.rules import RulesModel
from fraudnet.pipeline import ModelBundle, Prepared, fit_gbdt, fit_gnn, prepare, train_bundle, with_split
from fraudnet.serving.alerts import AlertQueue
from fraudnet.serving.app import create_app
from fraudnet.serving.service import ScoringService, Transaction, to_iso
from fraudnet.sim.config import DAY, HOUR, PATTERN_NAMES, SimConfig

logger = logging.getLogger("fraudnet.bench")
BENCH_DIR = ROOT / "benchmarks"
N_BOOT = 200
LATENCY_TXNS = 2000
HTTP_TXNS = 500
WARMUP_TXNS = 50
ALERT_BUDGET = 50


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
        )
        return out.stdout.strip() or "uncommitted"
    except OSError:
        return "unknown"


def _r(value: float, digits: int = 4) -> float:
    return round(float(value), digits)


def _curve(curve: list[tuple[int, float]]) -> list[list[float]]:
    """Validation average precision by tree count or epoch, for the report."""
    return [[int(step), _r(ap)] for step, ap in curve]


# Model evaluation -------------------------------------------------------------------------------------------


def _per_pattern(prep: Prepared, score: np.ndarray, threshold: float) -> dict[str, dict[str, float | int]]:
    """Transaction recall and attack detection rate per planted pattern at one threshold."""
    te = prep.masks["test"]
    pattern = prep.dataset.txns["pattern"][: te.size][te]
    attack = prep.dataset.txns["attack_id"][: te.size][te]
    flagged = score > threshold
    out: dict[str, dict[str, float | int]] = {}
    for code, name in PATTERN_NAMES.items():
        mask = pattern == code
        attacks = np.unique(attack[mask])
        caught = np.unique(attack[mask & flagged])
        out[name] = {
            "transactions": int(mask.sum()),
            "recall": _r(flagged[mask].mean()) if mask.any() else float("nan"),
            "attacks": int(attacks.size),
            "attacks_detected": _r(caught.size / attacks.size) if attacks.size else float("nan"),
        }
    return out


def _money(prep: Prepared, val_score: np.ndarray, test_score: np.ndarray, cost: CostModel) -> dict[str, Any]:
    """Money saved on test two ways: at the validation-chosen threshold and with a fixed daily alert budget."""
    va, te = prep.masks["val"], prep.masks["test"]
    threshold = cost.best_threshold(prep.labels["val"][va], prep.amount[va], val_score)
    y, amount = prep.labels["test"][te], prep.amount[te]
    day = prep.dataset.day[: te.size][te]
    loss = cost.loss_without_model(y, amount)

    def summary(alert: np.ndarray) -> dict[str, float]:
        saved = cost.money_saved(y, amount, alert)
        return {
            "alerts_per_day": _r(alert.sum() / max(1, np.unique(day).size), 1),
            "alert_precision": _r((y * alert).sum() / max(1, alert.sum())),
            "fraud_recall": _r((y * alert).sum() / max(1, y.sum())),
            "money_saved": round(saved, 2),
            "share_of_loss_saved": _r(saved / loss) if loss > 0 else float("nan"),
        }

    return {
        "threshold": float(threshold),
        **summary((test_score >= threshold).astype(np.int64)),
        f"budget_{ALERT_BUDGET}_per_day": summary(top_k_per_day(day, test_score, ALERT_BUDGET)),
        "loss_without_model": round(loss, 2),
        "oracle_saved": round(loss - cost.review_cost * float(y.sum()), 2),
    }


def evaluate(prep: Prepared, val_score: np.ndarray, test_score: np.ndarray, seed: int) -> dict[str, Any]:
    """All test metrics of one model, with day-block bootstrap intervals."""
    te = prep.masks["test"]
    y, day = prep.labels["test"][te], prep.dataset.day[: te.size][te]
    metrics = detection_metrics(y, test_score, day)
    ci = {
        "pr_auc": day_block_bootstrap(
            day,
            lambda i: average_precision_score(y[i], test_score[i]) if y[i].any() else float("nan"),
            N_BOOT,
            seed,
        ),
        "recall_at_1pct_fpr": day_block_bootstrap(
            day, lambda i: recall_at_fpr(y[i], test_score[i], 0.01), N_BOOT, seed
        ),
    }
    threshold = threshold_at_fpr(y, test_score, 0.01)
    return {
        "metrics": {k: _r(v) for k, v in metrics.items()},
        "ci95": {k: [_r(lo), _r(hi)] for k, (lo, hi) in ci.items()},
        "per_pattern_at_1pct_fpr": _per_pattern(prep, test_score, threshold),
        "money": _money(prep, val_score, test_score, CostModel()),
    }


def main_models(
    prep: Prepared, bundle: ModelBundle, seed: int
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Evaluate the rules, both boosted baselines, the GNN and the GBDT plus GNN average."""
    X, va, te = prep.table.X, prep.masks["val"], prep.masks["test"]
    scores: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "rules": (RulesModel().score(X[va]), RulesModel().score(X[te]))
    }
    fits: dict[str, dict[str, Any]] = {"rules": {"fit_seconds": 0.0}}
    for name, groups in (
        ("gbdt_tabular", TABULAR_GROUPS),
        (
            "gbdt_streaming",
            tuple(
                g for g in (*TABULAR_GROUPS, *GRAPH_GROUPS) if g not in ("degree", "pagerank", "community")
            ),
        ),
    ):
        started = time.perf_counter()
        model = fit_gbdt(prep, columns_for(groups), name)
        fits[name] = {
            "fit_seconds": _r(time.perf_counter() - started, 2),
            "trees": model.n_iter,
            "features": len(model.columns),
            "val_curve": _curve(model.val_curve),
        }
        scores[name] = (model.predict(X[va]), model.predict(X[te]))
    scores["gbdt"] = (bundle.gbdt.predict(X[va]), bundle.gbdt.predict(X[te]))
    fits["gbdt"] = {
        "fit_seconds": bundle.fit_seconds["gbdt"],
        "trees": bundle.gbdt.n_iter,
        "features": N_FEATURES,
        "val_curve": _curve(bundle.gbdt.val_curve),
    }
    if bundle.gnn is not None:
        scores["graphsage"] = (bundle.gnn.predict(prep.table, va), bundle.gnn.predict(prep.table, te))
        fits["graphsage"] = {
            "fit_seconds": bundle.fit_seconds["graphsage"],
            "best_epoch": bundle.gnn.best_epoch,
            "val_curve": _curve(bundle.gnn.val_curve),
        }
        scores["gbdt+graphsage"] = tuple(
            (a + b) / 2.0 for a, b in zip(scores["gbdt"], scores["graphsage"], strict=True)
        )  # type: ignore[assignment]
        fits["gbdt+graphsage"] = {
            "fit_seconds": _r(fits["gbdt"]["fit_seconds"] + fits["graphsage"]["fit_seconds"], 2)
        }
    results = {name: {**evaluate(prep, v, t, seed), "fit": fits[name]} for name, (v, t) in scores.items()}
    return results, {name: t for name, (_, t) in scores.items()}


# Ablation ---------------------------------------------------------------------------------------------------


def ablation(prep: Prepared, full_test_score: np.ndarray) -> dict[str, Any]:
    """Boosted models with graph feature groups added to the tabular base one at a time, or removed from all."""
    te = prep.masks["test"]
    y = prep.labels["test"][te]
    variants: dict[str, tuple[str, ...]] = {"tabular": TABULAR_GROUPS}
    variants |= {f"tabular+{g}": (*TABULAR_GROUPS, g) for g in GRAPH_GROUPS}
    variants |= {
        f"all-{g}": tuple(x for x in (*TABULAR_GROUPS, *GRAPH_GROUPS) if x != g) for g in GRAPH_GROUPS
    }
    out: dict[str, Any] = {}
    scores: dict[str, np.ndarray] = {}
    for name, groups in variants.items():
        scores[name] = fit_gbdt(prep, columns_for(groups), name).predict(prep.table.X[te])
        out[name] = _ablation_row(prep, y, scores[name], groups)
    scores["all"] = full_test_score
    out["all"] = _ablation_row(prep, y, full_test_score, (*TABULAR_GROUPS, *GRAPH_GROUPS))
    day = prep.dataset.day[: te.size][te]
    for name, row in out.items():
        row["delta_pr_auc_vs_tabular"] = _r(row["pr_auc"] - out["tabular"]["pr_auc"])
        row["delta_pr_auc_vs_all"] = _r(row["pr_auc"] - out["all"]["pr_auc"])
        lo, hi = (
            paired_delta_bootstrap(day, y, scores[name], scores["tabular"], N_BOOT, 0)
            if name != "tabular"
            else (0.0, 0.0)
        )
        row["delta_vs_tabular_ci95"] = [_r(lo), _r(hi)]
    return out


def _ablation_row(
    prep: Prepared, y: np.ndarray, score: np.ndarray, groups: tuple[str, ...]
) -> dict[str, Any]:
    threshold = threshold_at_fpr(y, score, 0.01)
    per = _per_pattern(prep, score, threshold)
    return {
        "groups": list(groups),
        "pr_auc": _r(average_precision_score(y, score)),
        "recall_at_1pct_fpr": _r(recall_at_fpr(y, score, 0.01)),
        "pattern_recall_at_1pct_fpr": {k: v["recall"] for k, v in per.items()},
    }


def protocol_ablation(prep: Prepared, results: dict[str, Any], seed: int) -> dict[str, Any]:
    """Refit without the label-maturity gap (validation up to the build day) and compare on the same test."""
    nogap = with_split(prep, TimeSplit.for_days(prep.split.n_days, gap=False))
    X, va, te = nogap.table.X, nogap.masks["val"], nogap.masks["test"]
    gbdt = fit_gbdt(nogap, list(range(N_FEATURES)), "gbdt_no_gap")
    gnn = fit_gnn(nogap)
    out: dict[str, Any] = {"split": nogap.split.describe(nogap.dataset)}
    for name, (val_score, test_score) in {
        "gbdt": (gbdt.predict(X[va]), gbdt.predict(X[te])),
        "graphsage": (gnn.predict(nogap.table, va), gnn.predict(nogap.table, te)),
    }.items():
        row = evaluate(nogap, val_score, test_score, seed)
        with_gap = results[name]
        out[name] = {
            "no_gap": _protocol_row(row),
            "with_gap": _protocol_row(with_gap),
        }
    return out


def _protocol_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "pr_auc": row["metrics"]["pr_auc"],
        "recall_at_1pct_fpr": row["metrics"]["recall_at_1pct_fpr"],
        "ring_recall_at_1pct_fpr": row["per_pattern_at_1pct_fpr"]["fraud_ring"]["recall"],
        "money_saved": row["money"]["money_saved"],
    }


# Point-in-time proof and the cost of leakage ----------------------------------------------------------------


def point_in_time(prep: Prepared, profile: str) -> dict[str, Any]:
    """Run the leakage checker on the first weeks of the real dataset, for the honest and a leaky pipeline."""
    horizon_days, cut_days = (30, 21) if profile == "full" else (20, 14)
    sub = known_at(prep.dataset, horizon_days * DAY)
    cut = cut_days * DAY + 7 * HOUR
    pipelines: dict[str, Callable] = {
        "honest": lambda ds: replay(ds).table.X,
        "leaky_instant_labels": lambda ds: replay(ds, schedule="instant").table.X,
    }
    out: dict[str, Any] = {"dataset_days": horizon_days, "cut_day": cut / DAY, "transactions": len(sub)}
    for name, pipeline in pipelines.items():
        started = time.perf_counter()
        reports = check_point_in_time(sub, cut, pipeline)
        out[name] = {
            "passed": all(r.passed for r in reports),
            "seconds": _r(time.perf_counter() - started, 1),
            "checks": [
                {
                    "variant": r.variant, "rows": r.rows_compared, "values": r.values_compared,
                    "mismatched_values": r.mismatched_values, "mismatched_features": list(r.mismatched_features),
                }
                for r in reports
            ],
        }  # fmt: skip
    return out


def leakage_cost(prep: Prepared, honest_test_score: np.ndarray) -> dict[str, Any]:
    """Train on leaky features (labels visible at transaction time) and compare offline with online reality.

    Offline, the leaky model is scored on leaky test features, which is what a careless evaluation reports.
    Online, the API can only compute honest features, so the same model meets honest features.
    """
    leaky = replay(prep.dataset, schedule="instant").table.X
    tr, va, te = prep.masks["train"], prep.masks["val"], prep.masks["test"]
    model = GBDTModel(list(range(N_FEATURES)), name="gbdt_leaky").fit(
        leaky[tr], prep.labels["train"][tr], leaky[va], prep.labels["val"][va]
    )
    y = prep.labels["test"][te]
    rows = {
        "leaky_model_on_leaky_features": model.predict(leaky[te]),
        "leaky_model_on_online_features": model.predict(prep.table.X[te]),
        "honest_model_on_online_features": honest_test_score,
    }
    return {
        name: {
            "pr_auc": _r(average_precision_score(y, s)),
            "recall_at_1pct_fpr": _r(recall_at_fpr(y, s, 0.01)),
        }
        for name, s in rows.items()
    }


# Latency ----------------------------------------------------------------------------------------------------


def _test_transactions(prep: Prepared, n: int) -> list[tuple[int, Transaction]]:
    t = prep.dataset.txns
    start = int(np.searchsorted(t["ts"], prep.split.build_ts, side="left"))
    rows = range(start, min(start + n, len(prep.dataset)))
    return [
        (
            i,
            Transaction(
                int(t["ts"][i]),
                int(t["account"][i]),
                int(t["device"][i]),
                int(t["merchant"][i]),
                int(t["dst"][i]),
                float(t["amount"][i]),
                int(t["channel"][i]),
                f"sim-{i}",
            ),
        )
        for i in rows
    ]


def _percentiles(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values)
    return {
        "p50_ms": _r(np.percentile(arr, 50), 3),
        "p95_ms": _r(np.percentile(arr, 95), 3),
        "p99_ms": _r(np.percentile(arr, 99), 3),
        "max_ms": _r(arr.max(), 3),
        "throughput_per_s": _r(1000.0 / arr.mean(), 1),
    }


def latency(prep: Prepared, bundle: ModelBundle) -> dict[str, Any]:
    """In-process latency per transaction (features, score, reason codes) and through HTTP; feature parity.

    The first ``WARMUP_TXNS`` test transactions are scored but not timed (lazy initialisation), and every
    path starts from the same pickled build-time store, so all of them see the same stream.
    """
    txns = _test_transactions(prep, WARMUP_TXNS + LATENCY_TXNS)
    out: dict[str, Any] = {"transactions": len(txns) - WARMUP_TXNS, "warmup": WARMUP_TXNS, "in_process": {}}
    parity = 0.0
    for model in bundle.models:
        for explain in (True, False):
            service = ScoringService(bundle, pickle.loads(prep.store_at_build), AlertQueue(), None, model)
            timings = []
            for k, (i, txn) in enumerate(txns):
                started = time.perf_counter()
                result = service.score(txn, model, explain)
                if k >= WARMUP_TXNS:
                    timings.append((time.perf_counter() - started) * 1000.0)
                if i < prep.table.X.shape[0]:
                    parity = max(parity, float(np.abs(result.features - prep.table.X[i]).max()))
            out["in_process"][f"{model}{'' if explain else '_no_reasons'}"] = _percentiles(timings)
    out["online_offline_feature_max_abs_diff"] = parity
    out["http"] = _http_latency(prep, bundle, txns[: WARMUP_TXNS + HTTP_TXNS])
    out["snapshot_refresh_s"] = _refresh_seconds(prep, bundle)
    return out


def _http_latency(
    prep: Prepared, bundle: ModelBundle, txns: list[tuple[int, Transaction]]
) -> dict[str, dict[str, float]]:
    service = ScoringService(bundle, pickle.loads(prep.store_at_build), AlertQueue(), None, "gbdt")
    settings = Settings.from_env()
    channels = {0: "pos", 1: "ecom", 2: "p2p"}
    timings: list[float] = []
    server: list[float] = []
    quiet = [logging.getLogger(name) for name in ("fraudnet.access", "httpx", "httpx2")]
    with TestClient(create_app(settings, service)) as client:
        for logger_ in quiet:
            logger_.setLevel(logging.WARNING)
        for k, (_, t) in enumerate(txns):
            body: dict[str, Any] = {
                "txn_id": t.txn_id,
                "timestamp": to_iso(t.ts),
                "account_id": t.account,
                "device_id": t.device,
                "amount": t.amount,
                "channel": channels[t.channel],
            }
            body |= {"merchant_id": t.merchant} if t.merchant >= 0 else {"payee_account_id": t.dst}
            started = time.perf_counter()
            response = client.post("/v1/score", json=body)
            if k >= WARMUP_TXNS:
                timings.append((time.perf_counter() - started) * 1000.0)
                server.append(float(response.headers["x-process-time-ms"]))
            response.raise_for_status()
    for logger_ in quiet:
        logger_.setLevel(logging.NOTSET)
    return {"client": _percentiles(timings), "server": _percentiles(server)}


def _refresh_seconds(prep: Prepared, bundle: ModelBundle) -> dict[str, float]:
    """Time one daily graph refresh (snapshot features plus GNN embeddings) on the build-time store."""
    store = pickle.loads(prep.store_at_build)
    started = time.perf_counter()
    store.rebuild_snapshot(prep.split.build_ts)
    graph = time.perf_counter() - started
    embed = 0.0
    if bundle.gnn is not None:
        started = time.perf_counter()
        bundle.gnn.embed_snapshot(GnnSnapshot.from_snapshot(store.snapshot))
        embed = time.perf_counter() - started
    return {"graph_features": _r(graph, 3), "gnn_embeddings": _r(embed, 3), "nodes": store.snapshot.n_nodes}


# Seeds ------------------------------------------------------------------------------------------------------


def profile_config(profile: str, seed: int) -> SimConfig:
    """The simulator config of a benchmark profile."""
    return {"full": SimConfig, "quick": SimConfig.quick, "small": SimConfig.small}[profile](seed=seed)


def seed_robustness(seeds: list[int], profile: str, data_dir: Path | None) -> dict[str, Any]:
    """Core models on other simulator seeds: does the ranking of models hold across worlds?"""
    out: dict[str, Any] = {}
    for seed in seeds:
        prep = prepare(profile_config(profile, seed), data_dir)
        X, te = prep.table.X, prep.masks["test"]
        y = prep.labels["test"][te]
        scores = {"rules": RulesModel().score(X[te])}
        scores["gbdt_tabular"] = fit_gbdt(prep, columns_for(TABULAR_GROUPS), "gbdt_tabular").predict(X[te])
        scores["gbdt"] = fit_gbdt(prep, list(range(N_FEATURES)), "gbdt").predict(X[te])
        scores["graphsage"] = fit_gnn(prep).predict(prep.table, te)
        out[str(seed)] = {
            name: {
                "pr_auc": _r(average_precision_score(y, s)),
                "recall_at_1pct_fpr": _r(recall_at_fpr(y, s, 0.01)),
            }
            for name, s in scores.items()
        }
        del prep
    return out


# Orchestration ----------------------------------------------------------------------------------------------


def _timed[T](timings: dict[str, float], name: str, fn: Callable[[], T]) -> T:
    started = time.perf_counter()
    value = fn()
    timings[name] = _r(time.perf_counter() - started, 1)
    logger.info("bench step done", extra={"step": name, "seconds": timings[name]})
    return value


def run(
    seed: int, profile: str, extra_seeds: list[int], data_dir: Path | None = ROOT / "data"
) -> dict[str, Any]:
    """The whole benchmark for one profile (full, quick or small); returns the results dictionary."""
    started = time.perf_counter()
    torch.set_num_threads(2)
    prep = prepare(profile_config(profile, seed), data_dir)
    timings: dict[str, float] = dict(prep.seconds)
    bundle = _timed(timings, "train_bundle", lambda: train_bundle(prep))
    results: dict[str, Any] = {
        "meta": _meta(seed, profile),
        "dataset": {**prep.dataset.summary(), "split": prep.split.describe(prep.dataset)},
    }
    results["models"], test_scores = _timed(timings, "models", lambda: main_models(prep, bundle, seed))
    results["ablation"] = _timed(timings, "ablation", lambda: ablation(prep, test_scores["gbdt"]))
    results["protocol"] = _timed(
        timings, "protocol", lambda: protocol_ablation(prep, results["models"], seed)
    )
    results["leakage_cost"] = _timed(timings, "leakage_cost", lambda: leakage_cost(prep, test_scores["gbdt"]))
    results["point_in_time"] = _timed(timings, "point_in_time", lambda: point_in_time(prep, profile))
    results["latency"] = _timed(timings, "latency", lambda: latency(prep, bundle))
    if extra_seeds:
        others = _timed(timings, "seeds", lambda: seed_robustness(extra_seeds, profile, data_dir))
        results["seeds"] = {str(seed): _seed_row(results), **others}
        results["seeds_summary"] = _seed_summary(results["seeds"])
    timings["total"] = _r(time.perf_counter() - started, 1)
    results["meta"]["seconds"] = timings
    return results


def _seed_row(results: dict[str, Any]) -> dict[str, Any]:
    keep = ("rules", "gbdt_tabular", "gbdt", "graphsage")
    return {
        k: {m: results["models"][k]["metrics"][m] for m in ("pr_auc", "recall_at_1pct_fpr")}
        for k in keep
        if k in results["models"]
    }


def _seed_summary(seeds: dict[str, dict[str, dict[str, float]]]) -> dict[str, dict[str, float]]:
    """Mean, sample standard deviation, minimum and maximum of each metric across seeds."""
    models = [m for m in next(iter(seeds.values())) if all(m in row for row in seeds.values())]
    out: dict[str, dict[str, float]] = {}
    for model in models:
        out[model] = {}
        for metric in ("pr_auc", "recall_at_1pct_fpr"):
            values = np.array([row[model][metric] for row in seeds.values()])
            out[model] |= {
                f"{metric}_mean": _r(values.mean()),
                f"{metric}_std": _r(values.std(ddof=1)) if values.size > 1 else 0.0,
                f"{metric}_min": _r(values.min()),
                f"{metric}_max": _r(values.max()),
            }
    return out


def _meta(seed: int, profile: str) -> dict[str, Any]:
    return {
        "seed": seed,
        "profile": profile,
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "commit": _git_commit(),
        "version": __version__,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "scikit_learn": sklearn.__version__,
        "numpy": np.__version__,
        "machine": platform.machine(),
        "threads": 2,
        "cost_model": {
            "review_cost": CostModel().review_cost,
            "false_alert_cost": CostModel().false_alert_cost,
            "chargeback_fee": CostModel().chargeback_fee,
        },
    }


def _parse(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m fraudnet.bench", description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--seeds", default="", help="extra simulator seeds for the robustness table, comma separated"
    )
    parser.add_argument("--quick", action="store_true", help="smaller CI world")
    parser.add_argument(
        "--out", type=Path, default=None, help="results JSON path (default benchmarks/results.json)"
    )
    parser.add_argument(
        "--check", action="store_true", help="compare results with benchmarks/thresholds.json"
    )
    parser.add_argument("--check-only", action="store_true", help="only check an existing results file")
    parser.add_argument(
        "--update-readme", action="store_true", help="copy the tables into README.md and REPORT.md"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = _parse(argv)
    configure_logging("INFO")
    out = args.out or BENCH_DIR / ("ci-results.json" if args.quick else "results.json")
    if not args.check_only:
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
        with threadpool_limits(limits=2):
            results = run(args.seed, "quick" if args.quick else "full", seeds)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2) + "\n")
        out.with_suffix(".md").write_text(render_markdown(results))
        print(f"wrote {out} and {out.with_suffix('.md')} in {results['meta']['seconds']['total']} s")
    results = json.loads(out.read_text())
    if args.update_readme:
        update_readme(ROOT / "README.md", results)
        if (ROOT / "REPORT.md").exists() and not args.quick:
            update_report(ROOT / "REPORT.md", results)
        print("README.md and REPORT.md tables updated")
    if args.check or args.check_only:
        failures = check_thresholds(results, BENCH_DIR / "thresholds.json")
        for failure in failures:
            print(f"THRESHOLD FAILED: {failure}")
        if failures:
            return 1
        print("all benchmark thresholds met")
    return 0


if __name__ == "__main__":
    sys.exit(main())
