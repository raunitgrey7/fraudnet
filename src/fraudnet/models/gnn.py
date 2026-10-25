"""A heterogeneous GraphSAGE fraud model written in plain PyTorch (no graph library).

Why a GNN: hand-made graph features (degree, PageRank, community size) summarise the neighbourhood with
formulas chosen in advance. A GraphSAGE layer instead learns how to aggregate neighbours: a device node mixes
the features of the accounts that used it, an account mixes its devices, merchants and transfer partners, and
two layers reach accounts that share a device with accounts that share a device with you.

How it is wired:

* Nodes are the accounts, devices and merchants of one daily snapshot, with features computed at snapshot
  time (``features.graph.node_features``). Three extra nodes stand for an account, device or merchant that
  is not in the snapshot yet (a device first seen today); they have no edges and learn a default embedding.
* Six relations (device from account, account from device, merchant from account, account from merchant,
  payer from payee, payee from payer), each with its own weight matrix and mean aggregation over a
  row-normalised sparse adjacency: ``h' = ReLU(LayerNorm(W_self h + sum_r W_r mean_{u in N_r} h_u))``.
* A transaction is scored from the embeddings of its account, device and counterparty concatenated with its
  streaming features (the same real-time features the boosted model sees, minus the batch graph columns the
  GNN replaces).

Training is full-batch per snapshot: for each training day, embed the day's snapshot and take one gradient
step on that day's transactions (all frauds plus a seeded share of genuine ones). The number of epochs is
chosen on the time-based validation window.
"""

from __future__ import annotations

import copy
import warnings
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn

from fraudnet.features.graph import NODE_FEATURES
from fraudnet.features.replay import FeatureTable, GnnSnapshot
from fraudnet.features.schema import CATEGORICAL_FEATURES, FEATURE_INDEX, STREAMING_GROUPS, columns_for
from fraudnet.sim.config import TRANSFER_CATEGORY

# torch flags CSR tensors as beta on every conversion; the operations used here (CSR times dense, with
# gradients for the dense side) are covered by tests/test_models.py.
warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta", category=UserWarning)

N_RELATIONS = 6
UNKNOWN_FLAG = 3
N_CHANNELS = 3
N_MERCHANT_CODES = TRANSFER_CATEGORY + 1


@dataclass(frozen=True)
class GNNConfig:
    """Hyperparameters; width and learning rate won the validation-only sweep in ``fraudnet.tune``."""

    hidden: int = 32
    layers: int = 2
    dropout: float = 0.1
    lr: float = 5e-3
    weight_decay: float = 1e-5
    epochs: int = 40
    patience: int = 6
    negative_rate: float = 0.3
    pos_weight: float = 5.0
    threads: int = 2
    seed: int = 0


@dataclass
class SnapTensors:
    """A snapshot as model inputs: node features plus one sparse adjacency per relation."""

    x: torch.Tensor
    adjs: list[torch.Tensor]
    n_nodes: int


def _row_normalised(rows: np.ndarray, cols: np.ndarray, n: int) -> torch.Tensor:
    """Sparse CSR matrix whose row ``i`` averages the neighbours of node ``i``."""
    if rows.size == 0:
        return torch.sparse_coo_tensor(
            torch.zeros((2, 0), dtype=torch.long), torch.zeros(0), (n, n)
        ).to_sparse_csr()
    degree = np.bincount(rows, minlength=n).astype(np.float32)
    values = torch.from_numpy(1.0 / degree[rows])
    index = torch.from_numpy(np.stack([rows, cols]).astype(np.int64))
    return torch.sparse_coo_tensor(index, values, (n, n), check_invariants=False).coalesce().to_sparse_csr()


def snapshot_tensors(snap: GnnSnapshot) -> SnapTensors:
    """Typed, row-normalised adjacencies and node features, with the three unknown-entity nodes appended."""
    n = snap.n_acct + snap.n_dev + snap.n_mer
    total = n + 3
    x = np.zeros((total, NODE_FEATURES), dtype=np.float32)
    x[:n] = snap.node_x
    for k in range(3):
        x[n + k, k] = 1.0
        x[n + k, UNKNOWN_FLAG] = 1.0
    acct, dev = snap.acct_dev[0].astype(np.int64), snap.acct_dev[1].astype(np.int64) + snap.n_acct
    buyer, mer = (
        snap.acct_mer[0].astype(np.int64),
        snap.acct_mer[1].astype(np.int64) + snap.n_acct + snap.n_dev,
    )
    src, dst = snap.transfers[0].astype(np.int64), snap.transfers[1].astype(np.int64)
    pairs = [(dev, acct), (acct, dev), (mer, buyer), (buyer, mer), (src, dst), (dst, src)]
    return SnapTensors(torch.from_numpy(x), [_row_normalised(r, c, total) for r, c in pairs], n)


class HeteroSAGELayer(nn.Module):
    """One GraphSAGE layer with a separate weight per relation and mean aggregation."""

    def __init__(self, dim_in: int, dim_out: int, n_relations: int = N_RELATIONS) -> None:
        super().__init__()
        self.self_lin = nn.Linear(dim_in, dim_out)
        self.rel_lins = nn.ModuleList(nn.Linear(dim_in, dim_out, bias=False) for _ in range(n_relations))
        self.norm = nn.LayerNorm(dim_out)

    def forward(self, h: torch.Tensor, adjs: list[torch.Tensor]) -> torch.Tensor:
        """Combine each node with the mean of its neighbours under every relation."""
        out = self.self_lin(h)
        for lin, adj in zip(self.rel_lins, adjs, strict=True):
            out = out + adj @ lin(h)
        return torch.relu(self.norm(out))


class FraudSAGE(nn.Module):
    """Node encoder (GraphSAGE layers) plus a transaction head over three embeddings and tabular input."""

    def __init__(self, tab_dim: int, config: GNNConfig) -> None:
        super().__init__()
        hidden = config.hidden
        self.inp = nn.Linear(NODE_FEATURES, hidden)
        self.layers = nn.ModuleList(HeteroSAGELayer(hidden, hidden) for _ in range(config.layers))
        self.tab = nn.Sequential(nn.Linear(tab_dim, 64), nn.ReLU())
        self.head = nn.Sequential(
            nn.Linear(3 * hidden + 64, 64), nn.ReLU(), nn.Dropout(config.dropout), nn.Linear(64, 1)
        )

    def embed(self, st: SnapTensors) -> torch.Tensor:
        """Embeddings of every node of a snapshot."""
        h = torch.relu(self.inp(st.x))
        for layer in self.layers:
            h = layer(h, st.adjs)
        return h

    def score(self, h: torch.Tensor, nodes: torch.Tensor, tab: torch.Tensor) -> torch.Tensor:
        """Logits for transactions given their (account, device, counterparty) node indices."""
        z = torch.cat([h[nodes[:, 0]], h[nodes[:, 1]], h[nodes[:, 2]], self.tab(tab)], dim=1)
        return self.head(z).squeeze(1)


class TabularEncoder:
    """Standardise the streaming features and one-hot the channel and merchant category."""

    def __init__(self) -> None:
        self.columns = [
            c
            for c in columns_for(STREAMING_GROUPS)
            if c not in {FEATURE_INDEX[n] for n in CATEGORICAL_FEATURES}
        ]
        self.mean = np.zeros(len(self.columns), dtype=np.float32)
        self.std = np.ones(len(self.columns), dtype=np.float32)

    @property
    def dim(self) -> int:
        """Width of the encoded vector."""
        return len(self.columns) + N_CHANNELS + N_MERCHANT_CODES

    def fit(self, X: np.ndarray) -> TabularEncoder:
        """Learn the scaling from training rows only."""
        sub = X[:, self.columns]
        self.mean = sub.mean(axis=0).astype(np.float32)
        self.std = np.maximum(sub.std(axis=0), 1e-3).astype(np.float32)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Encoded float32 matrix."""
        scaled = np.clip((X[:, self.columns] - self.mean) / self.std, -10.0, 10.0)
        channel = np.eye(N_CHANNELS, dtype=np.float32)[X[:, FEATURE_INDEX["channel"]].astype(int)]
        category = np.eye(N_MERCHANT_CODES, dtype=np.float32)[
            X[:, FEATURE_INDEX["merchant_category"]].astype(int)
        ]
        return np.concatenate([scaled.astype(np.float32), channel, category], axis=1)


def resolve_nodes(nodes: np.ndarray, X: np.ndarray, n_nodes: int) -> np.ndarray:
    """Replace -1 (entity absent from the snapshot) with the matching unknown-entity node."""
    transfer = X[:, FEATURE_INDEX["merchant_category"]] == TRANSFER_CATEGORY
    out = nodes.astype(np.int64).copy()
    out[:, 0] = np.where(out[:, 0] >= 0, out[:, 0], n_nodes)
    out[:, 1] = np.where(out[:, 1] >= 0, out[:, 1], n_nodes + 1)
    out[:, 2] = np.where(out[:, 2] >= 0, out[:, 2], np.where(transfer, n_nodes, n_nodes + 2))
    return out


def _by_snapshot(snap_id: np.ndarray, mask: np.ndarray) -> dict[int, np.ndarray]:
    rows = np.flatnonzero(mask)
    order = np.argsort(snap_id[rows], kind="stable")
    rows = rows[order]
    keys, starts = np.unique(snap_id[rows], return_index=True)
    bounds = [*starts[1:], rows.size]
    return {int(k): rows[s:e] for k, s, e in zip(keys, starts, bounds, strict=True)}


class GNNModel:
    """Training, batch inference and online scoring around ``FraudSAGE``."""

    name = "graphsage"

    def __init__(self, config: GNNConfig | None = None) -> None:
        self.config = config or GNNConfig()
        self.encoder = TabularEncoder()
        self.net: FraudSAGE | None = None
        self.best_epoch = 0
        self.val_curve: list[tuple[int, float]] = []

    def _network(self) -> FraudSAGE:
        if self.net is None:
            raise RuntimeError("model is not fitted")
        return self.net

    @staticmethod
    def _tensors(table: FeatureTable, cache: dict[int, SnapTensors], sid: int) -> SnapTensors:
        if sid not in cache:
            cache[sid] = snapshot_tensors(table.snapshots[sid])
        return cache[sid]

    def fit(self, table: FeatureTable, y: np.ndarray, train: np.ndarray, val: np.ndarray) -> GNNModel:
        """Train on the train window and keep the epoch with the best validation average precision."""
        cfg = self.config
        torch.manual_seed(cfg.seed)
        torch.set_num_threads(cfg.threads)
        rng = np.random.default_rng(cfg.seed)
        self.encoder.fit(table.X[train])
        tab = self.encoder.transform(table.X)
        self.net = FraudSAGE(self.encoder.dim, cfg)
        opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        groups, val_groups = _by_snapshot(table.snap_id, train), _by_snapshot(table.snap_id, val)
        val_rows = np.flatnonzero(val)
        cache: dict[int, SnapTensors] = {}
        best_ap, best_state, stale = -1.0, copy.deepcopy(self.net.state_dict()), 0
        for epoch in range(1, cfg.epochs + 1):
            self._epoch(table, tab, y, groups, cache, opt, rng)
            scores = self._predict_groups(table, tab, val_groups, cache)
            ap = float(average_precision_score(y[val_rows], scores[val_rows]))
            self.val_curve.append((epoch, ap))
            if ap > best_ap + 1e-4:
                best_ap, best_state, stale, self.best_epoch = (
                    ap,
                    copy.deepcopy(self.net.state_dict()),
                    0,
                    epoch,
                )
            else:
                stale += 1
                if stale >= cfg.patience:
                    break
        self.net.load_state_dict(best_state)
        return self

    def _epoch(
        self,
        table: FeatureTable,
        tab: np.ndarray,
        y: np.ndarray,
        groups: dict[int, np.ndarray],
        cache: dict[int, SnapTensors],
        opt: torch.optim.Optimizer,
        rng: np.random.Generator,
    ) -> None:
        """One pass over the training days in a seeded random order, one gradient step per day."""
        cfg, net = self.config, self._network()
        net.train()
        loss_fn = nn.BCEWithLogitsLoss(reduction="none", pos_weight=torch.tensor(cfg.pos_weight))
        for sid in rng.permutation(np.array(sorted(groups))):
            rows = groups[int(sid)]
            rows = rows[(y[rows] == 1) | (rng.random(rows.size) < cfg.negative_rate)]
            if rows.size == 0:
                continue
            st = self._tensors(table, cache, int(sid))
            nodes = torch.from_numpy(resolve_nodes(table.nodes[rows], table.X[rows], st.n_nodes))
            logits = net.score(net.embed(st), nodes, torch.from_numpy(tab[rows]))
            target = torch.from_numpy(y[rows].astype(np.float32))
            weight = torch.where(target > 0, 1.0, 1.0 / cfg.negative_rate)
            loss = (loss_fn(logits, target) * weight).sum() / weight.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()

    def _predict_groups(
        self,
        table: FeatureTable,
        tab: np.ndarray,
        groups: dict[int, np.ndarray],
        cache: dict[int, SnapTensors] | None = None,
    ) -> np.ndarray:
        net = self._network()
        net.eval()
        out = np.full(table.X.shape[0], np.nan, dtype=np.float64)
        with torch.no_grad():
            for sid, rows in groups.items():
                st = (
                    self._tensors(table, cache, sid)
                    if cache is not None
                    else snapshot_tensors(table.snapshots[sid])
                )
                nodes = torch.from_numpy(resolve_nodes(table.nodes[rows], table.X[rows], st.n_nodes))
                out[rows] = torch.sigmoid(
                    net.score(net.embed(st), nodes, torch.from_numpy(tab[rows]))
                ).numpy()
        return out

    def predict(self, table: FeatureTable, mask: np.ndarray) -> np.ndarray:
        """Fraud scores for the rows selected by ``mask`` (in row order)."""
        torch.set_num_threads(self.config.threads)
        tab = self.encoder.transform(table.X[mask])
        full_tab = np.zeros((table.X.shape[0], tab.shape[1]), dtype=np.float32)
        full_tab[mask] = tab
        scores = self._predict_groups(table, full_tab, _by_snapshot(table.snap_id, mask))
        return scores[mask]

    def embed_snapshot(self, snap: GnnSnapshot) -> tuple[torch.Tensor, int]:
        """Node embeddings of a fresh snapshot, computed once per refresh for online scoring."""
        st = snapshot_tensors(snap)
        net = self._network()
        net.eval()
        with torch.no_grad():
            return net.embed(st), st.n_nodes

    def score_rows(self, h: torch.Tensor, n_nodes: int, nodes: np.ndarray, X_rows: np.ndarray) -> np.ndarray:
        """Score transactions against precomputed embeddings (the online path)."""
        net = self._network()
        resolved = torch.from_numpy(resolve_nodes(nodes, X_rows, n_nodes))
        with torch.no_grad():
            logits = net.score(h, resolved, torch.from_numpy(self.encoder.transform(X_rows)))
        return torch.sigmoid(logits).numpy().astype(np.float64)
