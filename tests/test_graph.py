"""Graph algorithms and snapshot construction."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from fraudnet.features.graph import (
    ACCOUNT_ROW_ZERO,
    NODE_FEATURES,
    WindowEdges,
    build_snapshot,
    empty_snapshot,
    label_propagation,
    pagerank,
    window_accounts,
)
from fraudnet.sim.config import DAY


def _undirected(n: int, edges: list[tuple[int, int]]) -> sp.csr_array:
    rows = [a for a, b in edges] + [b for a, b in edges]
    cols = [b for a, b in edges] + [a for a, b in edges]
    return sp.csr_array((np.ones(len(rows)), (rows, cols)), shape=(n, n))


def test_pagerank_is_uniform_on_a_cycle_and_sums_to_one():
    rank = pagerank(_undirected(6, [(i, (i + 1) % 6) for i in range(6)]))
    assert rank.sum() == pytest.approx(1.0)
    assert np.allclose(rank, 1 / 6)


def test_pagerank_favours_the_hub_and_handles_isolated_nodes():
    rank = pagerank(_undirected(6, [(0, 1), (0, 2), (0, 3), (0, 4)]))
    assert rank.argmax() == 0
    assert rank[5] > 0 and rank.sum() == pytest.approx(1.0)


def test_personalised_pagerank_concentrates_near_the_seed():
    adj = _undirected(8, [(0, 1), (1, 2), (2, 3), (4, 5), (5, 6), (6, 7)])
    seed = np.zeros(8)
    seed[0] = 1.0
    risk = pagerank(adj, seed)
    assert risk[1] > risk[3] > 0
    assert np.all(risk[4:] == 0)


def test_label_propagation_separates_disconnected_cliques_deterministically():
    edges = [(a, b) for a in range(4) for b in range(a + 1, 4)] + [
        (a, b) for a in range(4, 8) for b in range(a + 1, 8)
    ]
    adj = _undirected(8, edges)
    first = label_propagation(adj, [np.arange(8)])
    second = label_propagation(adj, [np.arange(8)])
    assert np.array_equal(first, second)
    assert len(set(first[:4])) == 1 and len(set(first[4:])) == 1 and first[0] != first[4]


def test_label_propagation_settles_on_a_bipartite_star():
    adj = _undirected(5, [(0, 4), (1, 4), (2, 4), (3, 4)])
    labels = label_propagation(adj, [np.arange(4), np.array([4])])
    assert len(set(labels)) == 1


def _window() -> WindowEdges:
    # accounts 1, 2, 3 share device 10; account 4 uses device 11 and pays account 1; account 5 buys alone
    ts = np.arange(7) * 100
    return WindowEdges(
        ts=ts,
        account=np.array([1, 2, 3, 4, 4, 5, 1]),
        device=np.array([10, 10, 10, 11, 11, 12, 10]),
        merchant=np.array([0, 0, 1, -1, 2, 2, 1]),
        dst=np.array([-1, -1, -1, 1, -1, -1, -1]),
        amount=np.array([10.0, 20.0, 30.0, 500.0, 5.0, 2.0, 15.0]),
    )


def test_snapshot_builds_communities_from_shared_devices():
    edges = _window()
    accounts = window_accounts(edges)
    snap = build_snapshot(DAY, edges, np.full(accounts.size, -DAY), {2}, set(), np.zeros(5, dtype=np.int8))
    assert list(snap.account_ids) == [1, 2, 3, 4, 5]
    assert snap.n_nodes == 5 + 3 + 3
    comm = snap.community[:5]
    assert comm[0] == comm[1] == comm[2], "accounts sharing device 10 form one community"
    assert comm[4] != comm[0]
    size, devices, known = snap.acct_rows[0][4], snap.acct_rows[0][5], snap.acct_rows[0][6]
    assert size == pytest.approx(np.log1p(4)) or size == pytest.approx(np.log1p(3))
    assert devices >= np.log1p(1) and known == pytest.approx(np.log1p(1))
    assert snap.node_x.shape == (snap.n_nodes, NODE_FEATURES)


def test_snapshot_lookup_and_node_indices():
    edges = _window()
    accounts = window_accounts(edges)
    snap = build_snapshot(DAY, edges, np.full(accounts.size, -DAY), set(), set(), np.zeros(5, dtype=np.int8))
    known = snap.lookup(1, 10, 0, -1)
    unseen = snap.lookup(99, 99, 99, -1)
    assert len(known) == len(unseen) == 15
    assert unseen == [0.0] * 15
    assert known[0] > 0
    a, d, cp = snap.nodes_of(4, 11, -1, 1)
    assert a == 3 and d == 5 + 1 and cp == 0
    assert snap.nodes_of(99, 99, 99, -1) == (-1, -1, -1)


def test_personalised_risk_is_zero_without_reports():
    edges = _window()
    accounts = window_accounts(edges)
    snap = build_snapshot(DAY, edges, np.full(accounts.size, -DAY), set(), set(), np.zeros(5, dtype=np.int8))
    assert all(row[3] == 0.0 for row in snap.acct_rows)


def test_empty_snapshot_returns_zero_features():
    snap = empty_snapshot(0)
    assert snap.n_nodes == 0
    assert snap.lookup(1, 1, 1, -1) == [0.0] * 15
    assert tuple(snap.acct_rows) == () and ACCOUNT_ROW_ZERO == (0.0,) * 9
