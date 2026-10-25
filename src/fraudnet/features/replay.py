"""Offline replay: push the whole log through the online store to build the training feature table.

Replaying, instead of computing features with a separate batch job, is what guarantees that the model is
trained on exactly the values the API will compute. The replay also records, for every transaction, which
graph snapshot it saw and where its account, device and counterparty sit in that snapshot, which is what the
GraphSAGE model needs. ``schedule="instant"`` reproduces the most common leak in fraud pipelines (labels
joined from a later extract and treated as known at transaction time) so the benchmark can measure it.
"""

from __future__ import annotations

import pickle
import time
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from fraudnet.features.graph import GraphSnapshot
from fraudnet.features.schema import N_FEATURES
from fraudnet.features.store import LabelEvent, OnlineFeatureStore, StoreConfig
from fraudnet.sim.config import NEVER, Pattern
from fraudnet.sim.dataset import Dataset

LabelSchedule = Literal["reported", "instant"]


@dataclass
class GnnSnapshot:
    """The parts of a graph snapshot the GNN trains on (no Python dicts, to keep memory small)."""

    time: int
    n_acct: int
    n_dev: int
    n_mer: int
    acct_dev: np.ndarray
    acct_mer: np.ndarray
    transfers: np.ndarray
    node_x: np.ndarray

    @classmethod
    def from_snapshot(cls, snap: GraphSnapshot) -> GnnSnapshot:
        """Drop the lookup tables of a full snapshot."""
        return cls(
            time=snap.time,
            n_acct=int(snap.account_ids.size),
            n_dev=int(snap.device_ids.size),
            n_mer=int(snap.merchant_ids.size),
            acct_dev=snap.acct_dev,
            acct_mer=snap.acct_mer,
            transfers=snap.transfers,
            node_x=snap.node_x,
        )


@dataclass
class FeatureTable:
    """Point-in-time features for every transaction plus the GNN's view of each one."""

    X: np.ndarray
    snap_id: np.ndarray
    nodes: np.ndarray
    snapshots: list[GnnSnapshot] = field(default_factory=list)
    seconds: float = 0.0


def label_events(dataset: Dataset, schedule: LabelSchedule = "reported") -> list[LabelEvent]:
    """Fraud reports as scheduled events; ``instant`` makes every eventual report visible right away."""
    t = dataset.txns
    fraud = (t["pattern"] > Pattern.LEGIT) & (t["report_ts"] < NEVER)
    idx = np.flatnonzero(fraud)
    report = t["report_ts"][idx] if schedule == "reported" else t["ts"][idx] + 1
    return [
        LabelEvent(int(r), int(i), int(a), int(d), int(m), int(b))
        for r, i, a, d, m, b in zip(
            report, idx, t["account"][idx], t["device"][idx], t["merchant"][idx], t["dst"][idx], strict=True
        )
    ]


def new_store(
    dataset: Dataset, config: StoreConfig | None = None, on_snapshot: object = None
) -> OnlineFeatureStore:
    """A store wired to the dataset's reference tables."""
    return OnlineFeatureStore(
        dataset.accounts["signup_ts"],
        dataset.merchants["category"],
        config,
        on_snapshot=on_snapshot,  # type: ignore[arg-type]
    )


@dataclass
class ReplayResult:
    """Features, the store at the end of the replay and, optionally, the pickled store at a checkpoint."""

    table: FeatureTable
    store: OnlineFeatureStore
    checkpoint: bytes | None = None


def replay(
    dataset: Dataset,
    config: StoreConfig | None = None,
    schedule: LabelSchedule = "reported",
    keep_snapshots: bool = False,
    stop_ts: int | None = None,
    checkpoint_ts: int | None = None,
) -> ReplayResult:
    """Replay transactions in time order, up to ``stop_ts`` when given.

    With ``checkpoint_ts`` the store is advanced to that time and pickled just before the first transaction
    at or after it, which is the serving state of a model built at that moment.
    """
    snapshots: list[GnnSnapshot] = []
    store = new_store(
        dataset,
        config,
        (lambda s: snapshots.append(GnnSnapshot.from_snapshot(s))) if keep_snapshots else None,
    )
    for event in label_events(dataset, schedule):
        store.schedule_label(event)
    t = dataset.txns
    n = len(dataset) if stop_ts is None else int(np.searchsorted(t["ts"], stop_ts, side="left"))
    X = np.zeros((n, N_FEATURES), dtype=np.float32)
    snap_id = np.zeros(n, dtype=np.int32)
    nodes = np.full((n, 3), -1, dtype=np.int32)
    cols = [t[c][:n].tolist() for c in ("ts", "account", "device", "merchant", "dst", "amount", "channel")]
    checkpoint_at = (
        int(np.searchsorted(t["ts"], checkpoint_ts, side="left")) if checkpoint_ts is not None else -1
    )
    checkpoint: bytes | None = None
    started = time.perf_counter()
    for i, (ts, acct, dev, mer, dst, amt, ch) in enumerate(zip(*cols, strict=True)):
        if i == checkpoint_at:
            checkpoint = _checkpoint(store, int(checkpoint_ts))  # type: ignore[arg-type]
        store.advance_to(ts)
        X[i] = store.features(ts, acct, dev, mer, dst, amt, ch)
        snap_id[i] = store.snapshots_built - 1
        nodes[i] = store.snapshot.nodes_of(acct, dev, mer, dst)
        store.ingest(ts, acct, dev, mer, dst, amt)
    if checkpoint_ts is not None and checkpoint is None:
        checkpoint = _checkpoint(store, int(checkpoint_ts))
    table = FeatureTable(
        X=X, snap_id=snap_id, nodes=nodes, snapshots=snapshots, seconds=time.perf_counter() - started
    )
    return ReplayResult(table, store, checkpoint)


def _checkpoint(store: OnlineFeatureStore, ts: int) -> bytes:
    """Advance to ``ts`` (applying reports and building the snapshot due then) and pickle the store."""
    store.advance_to(ts)
    return pickle.dumps(store, protocol=pickle.HIGHEST_PROTOCOL)
