"""The online scoring path: point-in-time features, a score, reason codes, a decision and an alert.

Reading a transaction's features and ingesting it happen under one lock, because they must be atomic: a
second request for the same device has to see the first one, and nothing may read state that already
contains the transaction being scored. Model scoring and explanation run outside the lock.

GraphSAGE embeddings are recomputed whenever the store builds a new snapshot (once a day of stream time),
so an online GNN score costs only the transaction head, not a pass over the graph. Creating a service pins
the process's BLAS and OpenMP pools to two threads.
"""

from __future__ import annotations

import threading
import time
from collections import Counter, OrderedDict
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
from threadpoolctl import threadpool_limits

from fraudnet.features.graph import GraphSnapshot
from fraudnet.features.replay import GnnSnapshot
from fraudnet.features.schema import FEATURE_NAMES
from fraudnet.features.store import LabelEvent, OnlineFeatureStore
from fraudnet.pipeline import ModelBundle
from fraudnet.serving.alerts import AlertQueue
from fraudnet.sim.dataset import Dataset

EPOCH = datetime(2026, 7, 1, tzinfo=UTC)
RECENT_LIMIT = 200_000
# Single-row tree prediction is dominated by OpenMP start-up when every core is offered: 1.37 ms per call
# with the default pool against 0.32 ms with two threads on the build machine (see REPORT.md).
SERVING_THREADS = 2


def to_iso(ts: int) -> str:
    """Simulation seconds to an ISO 8601 UTC timestamp."""
    return (EPOCH + timedelta(seconds=int(ts))).strftime("%Y-%m-%dT%H:%M:%SZ")


def from_datetime(value: datetime) -> int:
    """A timestamp (naive values are taken as UTC) to simulation seconds."""
    aware = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    return int((aware - EPOCH).total_seconds())


class OutOfOrderError(ValueError):
    """A transaction older than the store clock: its features could no longer be point-in-time."""


@dataclass(frozen=True)
class Transaction:
    """One payment as the scoring path sees it."""

    ts: int
    account: int
    device: int
    merchant: int
    dst: int
    amount: float
    channel: int
    txn_id: str
    true_fraud: int | None = None


@dataclass
class ScoreResult:
    """Everything known about one scored transaction."""

    txn: Transaction
    model: str
    score: float
    decision: str
    reasons: list[dict[str, Any]]
    features: np.ndarray
    latency_ms: float
    alert_id: int | None


class ScoringService:
    """Wraps the feature store, the models and the alert queue behind one thread-safe interface."""

    def __init__(
        self,
        bundle: ModelBundle,
        store: OnlineFeatureStore,
        alerts: AlertQueue | None = None,
        feed: Dataset | None = None,
        default_model: str = "gbdt",
    ) -> None:
        self.bundle = bundle
        self.store = store
        self.alerts = alerts
        self.default_model = default_model if default_model in bundle.models else "gbdt"
        self.lock = threading.Lock()
        self.store.on_snapshot = self._on_snapshot
        self.gnn_state = self._embed(store.snapshot)
        self.last_score: dict[int, float] = {}
        self.recent: OrderedDict[str, Transaction] = OrderedDict()
        self.feed = feed
        self.feed_pos = (
            int(np.searchsorted(feed.txns["ts"], store.clock, side="left")) if feed is not None else 0
        )
        self.counts: Counter[str] = Counter()
        self.started = time.time()
        self._next_id = 0
        threadpool_limits(limits=SERVING_THREADS)

    # Snapshots and state -----------------------------------------------------------------------------------

    def _embed(self, snap: GraphSnapshot) -> tuple[Any, int] | None:
        if self.bundle.gnn is None:
            return None
        return self.bundle.gnn.embed_snapshot(GnnSnapshot.from_snapshot(snap))

    def _on_snapshot(self, snap: GraphSnapshot) -> None:
        self.gnn_state = self._embed(snap)
        self.counts["snapshots"] += 1

    def new_txn_id(self) -> str:
        """Sequential id for transactions submitted without one."""
        with self.lock:
            self._next_id += 1
            return f"txn-{self._next_id}"

    def _remember(self, txn: Transaction) -> None:
        self.recent[txn.txn_id] = txn
        if len(self.recent) > RECENT_LIMIT:
            self.recent.popitem(last=False)

    def _features(self, txn: Transaction) -> tuple[np.ndarray, tuple[int, int, int], tuple[Any, int] | None]:
        """Read point-in-time features and ingest, atomically."""
        with self.lock:
            if txn.ts < self.store.clock:
                raise OutOfOrderError(
                    f"timestamp {to_iso(txn.ts)} is older than the store clock {to_iso(self.store.clock)}"
                )
            self.store.advance_to(txn.ts)
            row = self.store.features(
                txn.ts, txn.account, txn.device, txn.merchant, txn.dst, txn.amount, txn.channel
            )
            nodes = self.store.snapshot.nodes_of(txn.account, txn.device, txn.merchant, txn.dst)
            gnn_state = self.gnn_state
            self.store.ingest(txn.ts, txn.account, txn.device, txn.merchant, txn.dst, txn.amount)
            self._remember(txn)
        return np.asarray(row, dtype=np.float32), nodes, gnn_state

    # Scoring -----------------------------------------------------------------------------------------------

    def _predict(
        self, model: str, rows: np.ndarray, nodes: tuple[int, int, int], gnn_state: tuple[Any, int] | None
    ) -> np.ndarray:
        if model == "gbdt":
            return self.bundle.gbdt.predict_fast(rows)
        if model == "graphsage" and self.bundle.gnn is not None and gnn_state is not None:
            h, n_nodes = gnn_state
            return self.bundle.gnn.score_rows(
                h, n_nodes, np.repeat(np.array([nodes]), rows.shape[0], axis=0), rows
            )
        return self.bundle.rules.score(rows)

    def decide(self, model: str, score: float) -> str:
        """Approve, review (an alert) or block, from the validation thresholds."""
        t = self.bundle.thresholds[model]
        if score >= t["block"]:
            return "block"
        return "review" if score >= t["review"] else "approve"

    def score(self, txn: Transaction, model: str | None = None, explain: bool = True) -> ScoreResult:
        """Score one transaction end to end; raises ``OutOfOrderError`` for stale timestamps."""
        model = model or self.default_model
        if model not in self.bundle.models:
            raise KeyError(f"unknown model {model}")
        started = time.perf_counter()
        row, nodes, gnn_state = self._features(txn)
        if model == "rules":
            score = float(self.bundle.rules.score(row[None, :])[0])
            reasons = self.bundle.rules.reasons(row) if explain else []
        elif explain:
            batch = self.bundle.explainer.variants(row)
            scores = self._predict(model, batch, nodes, gnn_state)
            score = float(scores[0])
            reasons = [r.to_dict() for r in self.bundle.explainer.reasons(row, scores)]
        else:
            score = float(self._predict(model, row[None, :], nodes, gnn_state)[0])
            reasons = []
        decision = self.decide(model, score)
        alert_id = self._alert(txn, model, score, decision, reasons) if decision != "approve" else None
        self.last_score[txn.account] = score
        self.counts[f"{model}:{decision}"] += 1
        latency = (time.perf_counter() - started) * 1000.0
        return ScoreResult(txn, model, score, decision, reasons, row, latency, alert_id)

    def _alert(
        self, txn: Transaction, model: str, score: float, decision: str, reasons: list[dict[str, Any]]
    ) -> int | None:
        if self.alerts is None:
            return None
        record = {**txn.__dict__, "model": model, "score": score, "decision": decision, "reasons": reasons}
        return self.alerts.add(record)

    # Labels ------------------------------------------------------------------------------------------------

    def report_fraud(self, txn_id: str) -> bool:
        """A chargeback or analyst verdict for a recent transaction, visible from the current clock on."""
        with self.lock:
            txn = self.recent.get(txn_id)
            if txn is None:
                return False
            event = LabelEvent(self.store.clock, -1, txn.account, txn.device, txn.merchant, txn.dst)
            self.store.schedule_label(event)
            self.counts["labels_reported"] += 1
        return True

    def resolve_alert(self, alert_id: int, label: str) -> dict[str, Any] | None:
        """Close an alert; a fraud verdict is fed back into the feature store."""
        if self.alerts is None:
            return None
        resolved = self.alerts.resolve(alert_id, "fraud" if label == "fraud" else "legit", self.store.clock)
        if resolved is not None and label == "fraud":
            self.report_fraud(resolved["txn_id"])
        return resolved

    # Simulator feed ----------------------------------------------------------------------------------------

    def replay_feed(self, n: int, model: str | None = None) -> dict[str, Any]:
        """Score the next ``n`` held-out simulator transactions through the normal path."""
        if self.feed is None:
            return {
                "scored": 0,
                "skipped": 0,
                "alerts": 0,
                "alerts_true_fraud": 0,
                "fraud_in_batch": 0,
                "remaining": 0,
            }
        t = self.feed.txns
        scored = skipped = alerts = hits = frauds = 0
        while scored + skipped < n and self.feed_pos < len(self.feed):
            i = self.feed_pos
            self.feed_pos += 1
            truth = int(t["pattern"][i] > 0)
            txn = Transaction(int(t["ts"][i]), int(t["account"][i]), int(t["device"][i]), int(t["merchant"][i]),
                              int(t["dst"][i]), float(t["amount"][i]), int(t["channel"][i]), f"sim-{i}", truth)  # fmt: skip
            try:
                result = self.score(txn, model)
            except OutOfOrderError:
                skipped += 1
                continue
            scored += 1
            frauds += truth
            if result.alert_id is not None:
                alerts += 1
                hits += truth
        return {
            "scored": scored, "skipped": skipped, "alerts": alerts, "alerts_true_fraud": hits,
            "fraud_in_batch": frauds, "remaining": len(self.feed) - self.feed_pos, "clock": to_iso(self.store.clock),
        }  # fmt: skip

    # Views -------------------------------------------------------------------------------------------------

    def rings(self, limit: int = 8, min_accounts: int = 3) -> list[dict[str, Any]]:
        """Communities of the latest snapshot that share devices, ranked by their members' latest scores."""
        with self.lock:
            snap = self.store.snapshot
            fraud_accounts = set(self.store.fraud_accounts)
        n_acct, n_dev = snap.account_ids.size, snap.device_ids.size
        if n_acct == 0:
            return []
        acct_comm = snap.community[:n_acct]
        dev_comm = snap.community[n_acct : n_acct + n_dev]
        sizes = np.bincount(acct_comm)
        out = []
        for comm in np.flatnonzero(sizes >= min_accounts):
            members = snap.account_ids[acct_comm == comm]
            devices = snap.device_ids[dev_comm == comm]
            edge_mask = np.isin(snap.acct_dev[0], np.flatnonzero(acct_comm == comm))
            shared = np.bincount(snap.acct_dev[1][edge_mask], minlength=n_dev)
            if devices.size == 0 or shared.max(initial=0) < 2:
                continue
            scores = [self.last_score[int(a)] for a in members if int(a) in self.last_score]
            out.append(self._ring_view(snap, int(comm), members, devices, edge_mask, scores, fraud_accounts))
        out.sort(key=lambda r: (-r["risk"], -r["accounts"]))
        return out[:limit]

    def _ring_view(
        self,
        snap: GraphSnapshot,
        comm: int,
        members: np.ndarray,
        devices: np.ndarray,
        edge_mask: np.ndarray,
        scores: list[float],
        fraud_accounts: set[int],
    ) -> dict[str, Any]:
        new = sum(1 for a in members if snap.time - self.store.signup_of(int(a), snap.time) < 30 * 86_400)
        edges = [
            {
                "source": f"a{int(snap.account_ids[a])}",
                "target": f"d{int(snap.device_ids[d])}",
                "kind": "device",
            }
            for a, d in zip(snap.acct_dev[0][edge_mask], snap.acct_dev[1][edge_mask], strict=True)
        ]
        nodes = [
            {
                "id": f"a{int(a)}",
                "type": "account",
                "score": self.last_score.get(int(a)),
                "reported_fraud": int(a) in fraud_accounts,
            }
            for a in members
        ]
        nodes += [{"id": f"d{int(d)}", "type": "device"} for d in devices]
        return {
            "community": comm,
            "accounts": int(members.size),
            "devices": int(devices.size),
            "new_accounts": new,
            "reported_fraud_accounts": sum(1 for a in members if int(a) in fraud_accounts),
            "risk": round(max(scores), 4) if scores else 0.0,
            "mean_score": round(float(np.mean(scores)), 4) if scores else None,
            "nodes": nodes,
            "edges": edges,
        }

    def network(self, account: int, max_neighbours: int = 15) -> dict[str, Any] | None:
        """Ego network of an account from live state: its devices, who else uses them, and its payees."""
        with self.lock:
            a = self.store.accounts.get(account)
            if a is None:
                return None
            nodes: dict[str, dict[str, Any]] = {f"a{account}": self._account_node(account)}
            edges: list[dict[str, str]] = []
            for dev in sorted(a.devices):
                d = self.store.devices[dev]
                nodes[f"d{dev}"] = {
                    "id": f"d{dev}",
                    "type": "device",
                    "reported_fraud": d.known_fraud > 0,
                    "accounts": len(d.accounts),
                }
                for other in sorted(d.accounts)[:max_neighbours]:
                    nodes.setdefault(f"a{other}", self._account_node(other))
                    edges.append({"source": f"a{other}", "target": f"d{dev}", "kind": "device"})
            for payee in sorted(a.payees)[:max_neighbours]:
                nodes.setdefault(f"a{payee}", self._account_node(payee))
                edges.append({"source": f"a{account}", "target": f"a{payee}", "kind": "transfer"})
            for _, sender in list(a.senders)[-max_neighbours:]:
                nodes.setdefault(f"a{sender}", self._account_node(sender))
                edges.append({"source": f"a{sender}", "target": f"a{account}", "kind": "transfer"})
        return {"account": account, "nodes": list(nodes.values()), "edges": edges}

    def _account_node(self, account: int) -> dict[str, Any]:
        return {
            "id": f"a{account}",
            "type": "account",
            "score": self.last_score.get(account),
            "reported_fraud": account in self.store.fraud_accounts,
        }

    def stats(self) -> dict[str, Any]:
        """Counters for the dashboard header and ``/v1/stats``."""
        with self.lock:
            store = {
                "clock": to_iso(self.store.clock) if self.store.clock > 0 else None,
                "accounts": len(self.store.accounts),
                "devices": len(self.store.devices),
                "merchants": len(self.store.merchants),
                "labels_applied": self.store.labels_applied,
                "labels_pending": len(self.store.pending),
                "snapshots_built": self.store.snapshots_built,
            }
        return {
            "store": store,
            "scored": dict(self.counts),
            "alerts": self.alerts.stats() if self.alerts is not None else {},
            "feed_remaining": (len(self.feed) - self.feed_pos) if self.feed is not None else 0,
            "features": len(FEATURE_NAMES),
        }
