"""Batch graph features computed on periodic snapshots of the payments graph.

A snapshot at time ``T`` is built only from transactions with timestamp before ``T`` (inside a trailing
window) and from fraud reports that arrived before ``T``. That is how a production graph job works: it runs
every few hours on the event log and its output is served until the next run, so a transaction can only
see the graph as it was at the last refresh. The snapshot holds three kinds of output:

* hand-crafted features (degree, PageRank, personalised PageRank from known fraud, community statistics)
  that the gradient boosting model reads,
* typed edge lists and node features that the GraphSAGE model aggregates over,
* community labels used by the ring view of the dashboard.

Communities come from deterministic label propagation over strong links only (shared devices and
transfers). Merchants are left out of the strong graph because one popular merchant would otherwise join
every customer into one community.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from fraudnet.sim.config import CATEGORIES, DAY

N_CATEGORIES = len(CATEGORIES)
NODE_FEATURES = 4 + 10 + 5 + N_CATEGORIES + 4  # type one-hot + unknown flag, account, device, merchant blocks
ACCOUNT_ROW_ZERO = (0.0,) * 9
DEVICE_ROW_ZERO = (0.0,) * 3
MERCHANT_ROW_ZERO = (0.0,) * 2
MERCHANT_EDGE_WEIGHT = 0.25
SMALL_AMOUNT = 5.0
NEW_ACCOUNT_DAYS = 30


@dataclass
class WindowEdges:
    """Transactions inside the snapshot window, as aligned arrays."""

    ts: np.ndarray
    account: np.ndarray
    device: np.ndarray
    merchant: np.ndarray
    dst: np.ndarray
    amount: np.ndarray


@dataclass
class GraphSnapshot:
    """The graph as of ``time`` plus everything derived from it."""

    time: int
    account_ids: np.ndarray
    device_ids: np.ndarray
    merchant_ids: np.ndarray
    acct_dev: np.ndarray
    acct_mer: np.ndarray
    transfers: np.ndarray
    node_x: np.ndarray
    community: np.ndarray
    acct_rows: list[tuple[float, ...]] = field(default_factory=list)
    dev_rows: list[tuple[float, ...]] = field(default_factory=list)
    mer_rows: list[tuple[float, ...]] = field(default_factory=list)
    acct_index: dict[int, int] = field(default_factory=dict)
    dev_index: dict[int, int] = field(default_factory=dict)
    mer_index: dict[int, int] = field(default_factory=dict)

    @property
    def n_nodes(self) -> int:
        """Accounts, devices and merchants in the window."""
        return int(self.account_ids.size + self.device_ids.size + self.merchant_ids.size)

    def lookup(self, account: int, device: int, merchant: int, dst: int) -> list[float]:
        """The 15 batch features of one transaction, in schema order (zeros for unseen entities)."""
        ai = self.acct_index.get(account)
        ar = self.acct_rows[ai] if ai is not None else ACCOUNT_ROW_ZERO
        di = self.dev_index.get(device)
        dr = self.dev_rows[di] if di is not None else DEVICE_ROW_ZERO
        if merchant >= 0:
            mi = self.mer_index.get(merchant)
            mr = self.mer_rows[mi] if mi is not None else MERCHANT_ROW_ZERO
            cp = (mr[0], mr[1], 0.0)
        else:
            bi = self.acct_index.get(dst)
            br = self.acct_rows[bi] if bi is not None else ACCOUNT_ROW_ZERO
            cp = (br[1], br[2], br[3])
        return [ar[0], ar[1], dr[0], cp[0], ar[2], dr[1], cp[1], ar[3], dr[2], cp[2], *ar[4:9]]

    def nodes_of(self, account: int, device: int, merchant: int, dst: int) -> tuple[int, int, int]:
        """Global node indices used by the GNN; -1 marks an entity absent from the snapshot."""
        n_acct, n_dev = self.account_ids.size, self.device_ids.size
        a = self.acct_index.get(account, -1)
        d = self.dev_index.get(device, -1)
        d = d + n_acct if d >= 0 else -1
        if merchant >= 0:
            m = self.mer_index.get(merchant, -1)
            cp = m + n_acct + n_dev if m >= 0 else -1
        else:
            cp = self.acct_index.get(dst, -1)
        return a, d, cp


def pagerank(
    adj: sp.csr_array, personalization: np.ndarray | None = None, damping: float = 0.85, iters: int = 60
) -> np.ndarray:
    """Power iteration on a weighted graph; dangling mass returns to the personalisation vector."""
    n = adj.shape[0]
    if n == 0:
        return np.zeros(0)
    out = np.asarray(adj.sum(axis=1)).ravel()
    inv = np.divide(1.0, out, out=np.zeros_like(out), where=out > 0)
    transition = (sp.diags_array(inv) @ adj).T.tocsr()
    v = np.full(n, 1.0 / n) if personalization is None else personalization / personalization.sum()
    rank = v.copy()
    dangling = out == 0
    for _ in range(iters):
        nxt = damping * (transition @ rank) + (damping * rank[dangling].sum() + 1.0 - damping) * v
        if np.abs(nxt - rank).sum() < 1e-10:
            return nxt
        rank = nxt
    return rank


def label_propagation(adj: sp.csr_array, blocks: list[np.ndarray], iters: int = 12) -> np.ndarray:
    """Deterministic label propagation, updating one node block at a time to avoid bipartite oscillation.

    Every node keeps a self-loop of weight one; ties go to the smallest label, so the result depends only on
    the graph and the node order.
    """
    n = adj.shape[0]
    weighted = (adj + sp.eye_array(n, format="csr")).tocsr()
    labels = np.arange(n)
    for _ in range(iters):
        changed = 0
        for block in blocks:
            if block.size == 0:
                continue
            rows = weighted[block]
            votes = sp.csr_array((rows.data, labels[rows.indices], rows.indptr), shape=(block.size, n))
            votes.sum_duplicates()
            new = np.asarray(votes.argmax(axis=1)).ravel()
            changed += int((new != labels[block]).sum())
            labels[block] = new
        if changed == 0:
            break
    return np.unique(labels, return_inverse=True)[1]


def _pairs(left: np.ndarray, right: np.ndarray, n_right: int) -> tuple[np.ndarray, np.ndarray]:
    """Unique (left, right) pairs and their multiplicity."""
    if left.size == 0:
        return np.zeros((2, 0), dtype=np.int64), np.zeros(0)
    keys, counts = np.unique(left.astype(np.int64) * n_right + right, return_counts=True)
    return np.vstack([keys // n_right, keys % n_right]), counts.astype(np.float64)


def _symmetric(n: int, edges: list[tuple[np.ndarray, np.ndarray, float]]) -> sp.csr_array:
    """Undirected weighted adjacency from (rows, cols, weight) edge blocks."""
    rows = np.concatenate([np.concatenate([r, c]) for r, c, _ in edges]) if edges else np.zeros(0, int)
    cols = np.concatenate([np.concatenate([c, r]) for r, c, _ in edges]) if edges else np.zeros(0, int)
    data = np.concatenate([np.full(2 * r.size, w) for r, _, w in edges]) if edges else np.zeros(0)
    return sp.csr_array((data, (rows, cols)), shape=(n, n))


@dataclass
class _Indexed:
    """Window edges mapped onto compact node indices."""

    accounts: np.ndarray
    devices: np.ndarray
    merchants: np.ndarray
    acc: np.ndarray
    dev: np.ndarray
    mer: np.ndarray
    dst: np.ndarray
    purchase: np.ndarray
    transfer: np.ndarray


def _index(edges: WindowEdges) -> _Indexed:
    transfer = edges.dst >= 0
    purchase = ~transfer
    accounts = np.unique(np.concatenate([edges.account, edges.dst[transfer]]))
    devices = np.unique(edges.device)
    merchants = np.unique(edges.merchant[purchase])
    return _Indexed(
        accounts=accounts,
        devices=devices,
        merchants=merchants,
        acc=np.searchsorted(accounts, edges.account),
        dev=np.searchsorted(devices, edges.device),
        mer=np.searchsorted(
            merchants, np.where(purchase, edges.merchant, merchants[0] if merchants.size else 0)
        ),
        dst=np.searchsorted(accounts, np.where(transfer, edges.dst, accounts[0])),
        purchase=purchase,
        transfer=transfer,
    )


def _community_rows(
    labels: np.ndarray, ix: _Indexed, dev_degree: np.ndarray, fraud_node: np.ndarray, new_acct: np.ndarray
) -> list[tuple]:
    """Per-account community statistics: size, devices, known fraud, new-account share, max device degree."""
    n_acct, n_dev = ix.accounts.size, ix.devices.size
    n_comm = int(labels.max()) + 1 if labels.size else 0
    acct_comm, dev_comm = labels[:n_acct], labels[n_acct : n_acct + n_dev]
    size = np.bincount(acct_comm, minlength=n_comm).astype(float)
    devices = np.bincount(dev_comm, minlength=n_comm).astype(float)
    fraud = np.bincount(labels[: n_acct + n_dev], weights=fraud_node[: n_acct + n_dev], minlength=n_comm)
    new = np.bincount(acct_comm, weights=new_acct, minlength=n_comm)
    max_deg = np.zeros(n_comm)
    np.maximum.at(max_deg, dev_comm, dev_degree)
    c = acct_comm
    stats = np.column_stack(
        [
            np.log1p(size[c]),
            np.log1p(devices[c]),
            np.log1p(fraud[c]),
            new[c] / np.maximum(size[c], 1),
            np.log1p(max_deg[c]),
        ]
    )
    return [tuple(float(v) for v in row) for row in stats]


def _fraud_nodes(ix: _Indexed, n: int, fraud_accounts: set[int], fraud_devices: set[int]) -> np.ndarray:
    """1.0 for accounts and devices with a fraud report known at snapshot time."""
    n_acct, n_dev = ix.accounts.size, ix.devices.size
    flag = np.zeros(n)
    flag[:n_acct] = np.isin(
        ix.accounts, np.fromiter(fraud_accounts, dtype=np.int64, count=len(fraud_accounts))
    )
    flag[n_acct : n_acct + n_dev] = np.isin(
        ix.devices, np.fromiter(fraud_devices, dtype=np.int64, count=len(fraud_devices))
    )
    return flag


def _entity_rows(
    ix: _Indexed, pairs: dict[str, np.ndarray], rank: np.ndarray, risk: np.ndarray, comm: list[tuple]
) -> tuple[list, list, list]:
    """Per-entity tuples of degree, PageRank, fraud proximity and community statistics."""
    n_acct, n_dev, n_mer = ix.accounts.size, ix.devices.size, ix.merchants.size
    acct_dev, acct_mer, transfers = pairs["acct_dev"], pairs["acct_mer"], pairs["transfers"]
    deg_m = np.log1p(np.bincount(acct_mer[0], minlength=n_acct))
    deg_cp = np.log1p(np.bincount(np.concatenate([transfers[0], transfers[1]]), minlength=n_acct))
    dev_deg = np.log1p(np.bincount(acct_dev[1], minlength=n_dev))
    mer_deg = np.log1p(np.bincount(acct_mer[1], minlength=n_mer))
    acct_rows = [
        (float(deg_m[i]), float(deg_cp[i]), float(rank[i]), float(risk[i]), *comm[i]) for i in range(n_acct)
    ]
    dev_rows = [(float(dev_deg[j]), float(rank[n_acct + j]), float(risk[n_acct + j])) for j in range(n_dev)]
    mer_rows = [(float(mer_deg[k]), float(rank[n_acct + n_dev + k])) for k in range(n_mer)]
    return acct_rows, dev_rows, mer_rows


def build_snapshot(
    time: int,
    edges: WindowEdges,
    signup_ts: np.ndarray,
    fraud_accounts: set[int],
    fraud_devices: set[int],
    merchant_category: np.ndarray,
) -> GraphSnapshot:
    """Build the snapshot at ``time`` from window edges and the fraud reports known at ``time``.

    ``signup_ts`` is aligned with the sorted unique accounts of the window (see ``window_accounts``).
    """
    if edges.ts.size == 0:
        return empty_snapshot(time)
    ix = _index(edges)
    n_acct, n_dev, n_mer = ix.accounts.size, ix.devices.size, ix.merchants.size
    n = n_acct + n_dev + n_mer
    pairs = {
        "acct_dev": _pairs(ix.acc, ix.dev, max(n_dev, 1))[0],
        "acct_mer": _pairs(ix.acc[ix.purchase], ix.mer[ix.purchase], max(n_mer, 1))[0],
        "transfers": _pairs(ix.acc[ix.transfer], ix.dst[ix.transfer], max(n_acct, 1))[0],
    }
    strong = [
        (pairs["acct_dev"][0], pairs["acct_dev"][1] + n_acct, 1.0),
        (pairs["transfers"][0], pairs["transfers"][1], 1.0),
    ]
    full = [*strong, (pairs["acct_mer"][0], pairs["acct_mer"][1] + n_acct + n_dev, MERCHANT_EDGE_WEIGHT)]
    g_strong = _symmetric(n, strong)
    fraud_node = _fraud_nodes(ix, n, fraud_accounts, fraud_devices)
    rank = np.log1p(pagerank(_symmetric(n, full)) * n)
    risk = np.log1p(pagerank(g_strong, fraud_node) * n) if fraud_node.any() else np.zeros(n)
    labels = label_propagation(g_strong, [np.arange(n_acct), np.arange(n_acct, n_acct + n_dev)])
    dev_degree = np.bincount(pairs["acct_dev"][1], minlength=n_dev).astype(float)
    new_acct = ((time - signup_ts) < NEW_ACCOUNT_DAYS * DAY).astype(float)
    comm = _community_rows(labels, ix, dev_degree, fraud_node, new_acct)
    acct_rows, dev_rows, mer_rows = _entity_rows(ix, pairs, rank, risk, comm)
    return GraphSnapshot(
        time=time,
        account_ids=ix.accounts,
        device_ids=ix.devices,
        merchant_ids=ix.merchants,
        acct_dev=pairs["acct_dev"].astype(np.int32),
        acct_mer=pairs["acct_mer"].astype(np.int32),
        transfers=pairs["transfers"].astype(np.int32),
        node_x=node_features(time, edges, ix, signup_ts, fraud_node, merchant_category),
        community=labels.astype(np.int32),
        acct_rows=acct_rows,
        dev_rows=dev_rows,
        mer_rows=mer_rows,
        acct_index={int(a): i for i, a in enumerate(ix.accounts)},
        dev_index={int(d): j for j, d in enumerate(ix.devices)},
        mer_index={int(m): k for k, m in enumerate(ix.merchants)},
    )


def empty_snapshot(time: int) -> GraphSnapshot:
    """The snapshot before any transaction has been seen."""
    none = np.zeros(0, dtype=np.int64)
    pairs = np.zeros((2, 0), dtype=np.int32)
    return GraphSnapshot(
        time=time,
        account_ids=none,
        device_ids=none,
        merchant_ids=none,
        acct_dev=pairs,
        acct_mer=pairs,
        transfers=pairs,
        node_x=np.zeros((0, NODE_FEATURES), dtype=np.float32),
        community=np.zeros(0, dtype=np.int32),
    )


def window_accounts(edges: WindowEdges) -> np.ndarray:
    """Sorted unique accounts of a window (senders and receivers), the order ``build_snapshot`` expects."""
    return np.unique(np.concatenate([edges.account, edges.dst[edges.dst >= 0]]))


def _mean_by(index: np.ndarray, values: np.ndarray, size: int) -> np.ndarray:
    count = np.bincount(index, minlength=size)
    return np.bincount(index, weights=values, minlength=size) / np.maximum(count, 1)


def node_features(
    time: int,
    edges: WindowEdges,
    ix: _Indexed,
    signup_ts: np.ndarray,
    fraud_node: np.ndarray,
    merchant_category: np.ndarray,
) -> np.ndarray:
    """Input features of every node for the GNN: type, activity, spend, degree and known-fraud flags."""
    n_acct, n_dev, n_mer = ix.accounts.size, ix.devices.size, ix.merchants.size
    x = np.zeros((n_acct + n_dev + n_mer, NODE_FEATURES), dtype=np.float32)
    log_amt = np.log1p(edges.amount)
    small = (edges.amount < SMALL_AMOUNT).astype(float)
    a = slice(0, n_acct)
    x[a, 0] = 1.0
    sent = np.bincount(ix.acc, minlength=n_acct)
    mean = _mean_by(ix.acc, log_amt, n_acct)
    var = _mean_by(ix.acc, log_amt**2, n_acct) - mean**2
    acct_dev, _ = _pairs(ix.acc, ix.dev, max(n_dev, 1))
    acct_mer, _ = _pairs(ix.acc[ix.purchase], ix.mer[ix.purchase], max(n_mer, 1))
    inflow = np.bincount(ix.dst[ix.transfer], weights=edges.amount[ix.transfer], minlength=n_acct)
    outflow = np.bincount(ix.acc[ix.transfer], weights=edges.amount[ix.transfer], minlength=n_acct)
    cps = np.bincount(np.concatenate([ix.acc[ix.transfer], ix.dst[ix.transfer]]), minlength=n_acct)
    x[a, 4:14] = np.column_stack(
        [
            np.log1p(sent), mean, np.sqrt(np.maximum(var, 0.0)),
            np.log1p(np.bincount(acct_dev[0], minlength=n_acct)), np.log1p(np.bincount(acct_mer[0], minlength=n_acct)),
            np.log1p(cps), np.log1p(inflow), np.log1p(outflow),
            np.log1p(np.maximum(time - signup_ts, 0) / DAY), fraud_node[:n_acct],
        ]
    )  # fmt: skip
    d = slice(n_acct, n_acct + n_dev)
    x[d, 1] = 1.0
    first_seen = np.full(n_dev, time, dtype=np.int64)
    np.minimum.at(first_seen, ix.dev, edges.ts)
    x[d, 14:19] = np.column_stack(
        [
            np.log1p(np.bincount(ix.dev, minlength=n_dev)), np.log1p(np.bincount(acct_dev[1], minlength=n_dev)),
            np.log1p((time - first_seen) / DAY), fraud_node[n_acct : n_acct + n_dev], _mean_by(ix.dev, small, n_dev),
        ]
    )  # fmt: skip
    m = slice(n_acct + n_dev, n_acct + n_dev + n_mer)
    x[m, 2] = 1.0
    if n_mer:
        x[m, 19:29] = np.eye(N_CATEGORIES, dtype=np.float32)[
            np.minimum(merchant_category[ix.merchants], N_CATEGORIES - 1)
        ]
        mer = ix.mer[ix.purchase]
        x[m, 29:33] = np.column_stack(
            [
                np.log1p(np.bincount(mer, minlength=n_mer)), np.log1p(np.bincount(acct_mer[1], minlength=n_mer)),
                _mean_by(mer, log_amt[ix.purchase], n_mer), _mean_by(mer, small[ix.purchase], n_mer),
            ]
        )  # fmt: skip
    return x
