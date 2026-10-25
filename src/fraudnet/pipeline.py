"""Shared steps of training and benchmarking: simulate, replay, split, fit, choose thresholds, bundle.

``fraudnet train`` and ``python -m fraudnet.bench`` call the same functions, so the model the API serves is
built exactly the way the benchmarked model was. Thresholds are chosen on the validation window with the
labels known at build time: the review threshold maximises money saved under the cost model, the block
threshold flags at most 0.1 percent of genuine validation transactions.
"""

from __future__ import annotations

import logging
import pickle
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from fraudnet import __version__
from fraudnet.evaluation.metrics import CostModel, threshold_at_fpr
from fraudnet.evaluation.split import TimeSplit
from fraudnet.features.replay import FeatureTable, replay
from fraudnet.features.schema import N_FEATURES
from fraudnet.features.store import OnlineFeatureStore
from fraudnet.models.explain import OcclusionExplainer
from fraudnet.models.gbdt import GBDTConfig, GBDTModel
from fraudnet.models.gnn import GNNConfig, GNNModel
from fraudnet.models.rules import RULE_ALERT_SCORE, RulesModel
from fraudnet.sim.config import SimConfig
from fraudnet.sim.dataset import Dataset
from fraudnet.sim.simulate import load_or_simulate

logger = logging.getLogger(__name__)
BLOCK_FPR = 0.001
BUNDLE_FILE = "bundle.pkl"
STORE_FILE = "store_at_build.pkl"


@dataclass
class Prepared:
    """A simulated world, its point-in-time features and the split, ready for model fitting."""

    dataset: Dataset
    split: TimeSplit
    table: FeatureTable
    masks: dict[str, np.ndarray]
    labels: dict[str, np.ndarray]
    store_at_build: bytes
    seconds: dict[str, float] = field(default_factory=dict)

    @property
    def amount(self) -> np.ndarray:
        """Transaction amounts aligned with the feature table."""
        return self.dataset.txns["amount"][: self.table.X.shape[0]]


def prepare(cfg: SimConfig, data_dir: Path | None, through_test: bool = True) -> Prepared:
    """Simulate (or load) the world and replay it; ``through_test=False`` stops at build time."""
    started = time.perf_counter()
    dataset = load_or_simulate(cfg, data_dir)
    simulated = time.perf_counter() - started
    split = TimeSplit.for_days(cfg.n_days, gap=True)
    result = replay(
        dataset,
        keep_snapshots=True,
        stop_ts=None if through_test else split.build_ts,
        checkpoint_ts=split.build_ts,
    )
    n = result.table.X.shape[0]
    masks = {k: v[:n] for k, v in split.masks(dataset.txns["ts"]).items()}
    labels = {k: v[:n] for k, v in split.labels(dataset).items()}
    if result.checkpoint is None:
        raise RuntimeError("replay did not reach the build time")
    seconds = {"simulate": round(simulated, 2), "replay": round(result.table.seconds, 2)}
    return Prepared(dataset, split, result.table, masks, labels, result.checkpoint, seconds)


def with_split(prep: Prepared, split: TimeSplit) -> Prepared:
    """The same replay under another split with the same build day (features do not depend on the split)."""
    if split.build_ts != prep.split.build_ts:
        raise ValueError("a different build day needs a different store checkpoint; call prepare again")
    n = prep.table.X.shape[0]
    masks = {k: v[:n] for k, v in split.masks(prep.dataset.txns["ts"]).items()}
    labels = {k: v[:n] for k, v in split.labels(prep.dataset).items()}
    return Prepared(prep.dataset, split, prep.table, masks, labels, prep.store_at_build, dict(prep.seconds))


@dataclass
class ModelBundle:
    """Everything the API needs to score: models, explainer, thresholds and provenance."""

    version: str
    sim_config: dict[str, float | int]
    split: TimeSplit
    rules: RulesModel
    gbdt: GBDTModel
    gnn: GNNModel | None
    explainer: OcclusionExplainer
    thresholds: dict[str, dict[str, float]]
    trained_at: str
    fit_seconds: dict[str, float]

    @property
    def models(self) -> list[str]:
        """Names of the models that can score."""
        return ["rules", "gbdt"] + (["graphsage"] if self.gnn is not None else [])


def validation_thresholds(
    y: np.ndarray, amount: np.ndarray, score: np.ndarray, cost: CostModel
) -> dict[str, float]:
    """Money-optimal review threshold and a 0.1 percent FPR block threshold, both from validation."""
    review = cost.best_threshold(y, amount, score)
    block = float(np.nextafter(threshold_at_fpr(y, score, BLOCK_FPR), np.inf))
    return {"review": review, "block": max(block, review)}


def fit_gbdt(
    prep: Prepared, columns: list[int], name: str, config: GBDTConfig | None = None, refit: bool = False
) -> GBDTModel:
    """Fit a boosted model on the train window, choosing its size on validation."""
    tr, va = prep.masks["train"], prep.masks["val"]
    X = prep.table.X
    return GBDTModel(columns, config, name).fit(
        X[tr], prep.labels["train"][tr], X[va], prep.labels["val"][va], refit=refit
    )


def fit_gnn(prep: Prepared, config: GNNConfig | None = None) -> GNNModel:
    """Fit the GraphSAGE model on the train window with validation early stopping."""
    return GNNModel(config).fit(prep.table, prep.labels["train"], prep.masks["train"], prep.masks["val"])


def train_bundle(
    prep: Prepared,
    gbdt_cfg: GBDTConfig | None = None,
    gnn_cfg: GNNConfig | None = None,
    with_gnn: bool = True,
) -> ModelBundle:
    """Fit the deployable models and choose their thresholds on validation."""
    started = time.perf_counter()
    gbdt = fit_gbdt(prep, list(range(N_FEATURES)), "gbdt", gbdt_cfg, refit=True)
    fit_seconds = {"gbdt": round(time.perf_counter() - started, 2)}
    gnn = None
    if with_gnn:
        started = time.perf_counter()
        gnn = fit_gnn(prep, gnn_cfg)
        fit_seconds["graphsage"] = round(time.perf_counter() - started, 2)
    tr, va = prep.masks["train"], prep.masks["val"]
    X, y_val, amount = prep.table.X, prep.labels["val"][va], prep.amount[va]
    cost = CostModel()
    rules = RulesModel()
    thresholds = {
        "rules": {"review": RULE_ALERT_SCORE, "block": RULE_ALERT_SCORE + 4.0},
        "gbdt": validation_thresholds(y_val, amount, gbdt.predict(X[va]), cost),
    }
    if gnn is not None:
        thresholds["graphsage"] = validation_thresholds(y_val, amount, gnn.predict(prep.table, va), cost)
    return ModelBundle(
        version=__version__,
        sim_config=prep.dataset.config.to_dict(),
        split=prep.split,
        rules=rules,
        gbdt=gbdt,
        gnn=gnn,
        explainer=OcclusionExplainer.from_training(X[tr], prep.labels["train"][tr]),
        thresholds=thresholds,
        trained_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        fit_seconds=fit_seconds,
    )


def save_artifacts(bundle: ModelBundle, store_at_build: bytes, artifact_dir: Path) -> None:
    """Write the bundle and the feature store as of build time."""
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / BUNDLE_FILE).write_bytes(pickle.dumps(bundle, protocol=pickle.HIGHEST_PROTOCOL))
    (artifact_dir / STORE_FILE).write_bytes(store_at_build)


def load_artifacts(artifact_dir: Path) -> tuple[ModelBundle, OnlineFeatureStore]:
    """Read what ``save_artifacts`` wrote (trusted local files produced by ``fraudnet train``)."""
    bundle = pickle.loads((artifact_dir / BUNDLE_FILE).read_bytes())
    store = pickle.loads((artifact_dir / STORE_FILE).read_bytes())
    return bundle, store


def artifacts_exist(artifact_dir: Path) -> bool:
    """True when a trained bundle and store are present."""
    return (artifact_dir / BUNDLE_FILE).exists() and (artifact_dir / STORE_FILE).exists()
