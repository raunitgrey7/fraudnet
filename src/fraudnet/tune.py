"""Validation-only hyperparameter sweep for the boosted model and the GNN.

``python -m fraudnet.tune`` fits a handful of configurations on the train window and reports their average
precision on the validation window, with the labels known at build time. Test labels are never read. The
defaults in ``GBDTConfig`` and ``GNNConfig`` are the winners of this sweep (output recorded in REPORT.md).
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace

import numpy as np
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits

from fraudnet.config import ROOT
from fraudnet.features.schema import N_FEATURES
from fraudnet.models.gbdt import GBDTConfig
from fraudnet.models.gnn import GNNConfig
from fraudnet.pipeline import Prepared, fit_gbdt, fit_gnn, prepare
from fraudnet.sim.config import SimConfig

GBDT_GRID = [{"learning_rate": lr, "max_leaf_nodes": leaves} for lr in (0.05, 0.1) for leaves in (15, 31, 63)]
GNN_GRID = [
    {"hidden": 48, "lr": 3e-3},
    {"hidden": 64, "lr": 2e-3},
    {"hidden": 32, "lr": 5e-3},
    {"hidden": 64, "lr": 2e-3, "negative_rate": 0.6},
]


def _val_ap(prep: Prepared, score: np.ndarray) -> float:
    return float(average_precision_score(prep.labels["val"][prep.masks["val"]], score))


def sweep(prep: Prepared) -> dict[str, list[dict[str, object]]]:
    """Fit every grid point and return validation AP and fit time for each."""
    out: dict[str, list[dict[str, object]]] = {"gbdt": [], "graphsage": []}
    va = prep.masks["val"]
    for params in GBDT_GRID:
        started = time.perf_counter()
        model = fit_gbdt(prep, list(range(N_FEATURES)), "gbdt", replace(GBDTConfig(), **params))
        out["gbdt"].append(
            {
                **params,
                "trees": model.n_iter,
                "val_ap": round(_val_ap(prep, model.predict(prep.table.X[va])), 4),
                "seconds": round(time.perf_counter() - started, 1),
            }
        )
        print("gbdt", out["gbdt"][-1], flush=True)
    for params in GNN_GRID:
        started = time.perf_counter()
        config = replace(GNNConfig(epochs=40, patience=6), **params)
        model = fit_gnn(prep, config)
        out["graphsage"].append(
            {
                **params,
                "best_epoch": model.best_epoch,
                "val_ap": round(_val_ap(prep, model.predict(prep.table, va)), 4),
                "seconds": round(time.perf_counter() - started, 1),
            }
        )
        print("graphsage", out["graphsage"][-1], flush=True)
    return out


def main(argv: list[str] | None = None) -> int:
    """Run the sweep on the full world (or the quick one) and print the winners."""
    parser = argparse.ArgumentParser(prog="python -m fraudnet.tune", description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args(argv)
    cfg = SimConfig.quick(args.seed) if args.quick else SimConfig(seed=args.seed)
    with threadpool_limits(limits=2):
        prep = prepare(cfg, ROOT / "data")
        results = sweep(prep)
    best = {name: max(rows, key=lambda r: r["val_ap"]) for name, rows in results.items()}
    print(
        json.dumps(
            {"best": best, "defaults": {"gbdt": asdict(GBDTConfig()), "graphsage": asdict(GNNConfig())}},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
