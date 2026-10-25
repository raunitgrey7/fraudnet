"""Shared fixtures: the small simulated world, its replay, a trained bundle and a scoring service factory."""

from __future__ import annotations

import pickle
from collections.abc import Callable

import numpy as np
import pytest
import torch

from fraudnet.features.replay import replay
from fraudnet.models.gnn import GNNConfig
from fraudnet.pipeline import ModelBundle, Prepared, prepare, train_bundle
from fraudnet.serving.alerts import AlertQueue
from fraudnet.serving.service import ScoringService, Transaction
from fraudnet.sim import Dataset, SimConfig, simulate
from fraudnet.sim.config import DAY

SMALL = SimConfig.small()


@pytest.fixture(autouse=True)
def _limit_threads() -> None:
    torch.set_num_threads(2)


@pytest.fixture(scope="session")
def small_dataset() -> Dataset:
    return simulate(SMALL)


@pytest.fixture(scope="session")
def prepared() -> Prepared:
    return prepare(SMALL, None)


@pytest.fixture(scope="session")
def bundle(prepared: Prepared) -> ModelBundle:
    return train_bundle(prepared, gnn_cfg=GNNConfig(epochs=4, patience=4))


@pytest.fixture(scope="session")
def first_month_features(small_dataset: Dataset) -> np.ndarray:
    return replay(small_dataset, stop_ts=30 * DAY).table.X


@pytest.fixture
def make_service(prepared: Prepared, bundle: ModelBundle) -> Callable[..., ScoringService]:
    def factory(model: str = "gbdt", feed: bool = False) -> ScoringService:
        store = pickle.loads(prepared.store_at_build)
        return ScoringService(bundle, store, AlertQueue(), prepared.dataset if feed else None, model)

    return factory


@pytest.fixture
def held_out(prepared: Prepared) -> Callable[[int], list[tuple[int, Transaction]]]:
    """The first ``n`` test-window transactions of the small world as service inputs, with their row index."""
    t = prepared.dataset.txns
    start = int(np.searchsorted(t["ts"], prepared.split.build_ts, side="left"))

    def take(n: int) -> list[tuple[int, Transaction]]:
        return [
            (
                i,
                Transaction(
                    int(t["ts"][i]), int(t["account"][i]), int(t["device"][i]), int(t["merchant"][i]),
                    int(t["dst"][i]), float(t["amount"][i]), int(t["channel"][i]), f"sim-{i}",
                    int(t["pattern"][i] > 0),
                ),
            )
            for i in range(start, start + n)
        ]  # fmt: skip

    return take
