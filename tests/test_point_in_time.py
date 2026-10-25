"""Proof of point-in-time correctness, and proof that the proof can fail.

The checker replays the log as it was known at a cut time and with a scrambled future; the honest pipeline
must produce bit-identical features for every transaction before the cut. Two deliberately leaky pipelines
(labels used before they were reported, and a statistic computed over the whole log) must be caught.
"""

from __future__ import annotations

import numpy as np
import pytest

from fraudnet.features.leakage import check_point_in_time, known_at, scrambled_future
from fraudnet.features.replay import replay
from fraudnet.features.schema import FEATURE_NAMES
from fraudnet.sim import Dataset
from fraudnet.sim.config import DAY, HOUR, NEVER


def honest(ds: Dataset) -> np.ndarray:
    return replay(ds).table.X


def instant_labels(ds: Dataset) -> np.ndarray:
    return replay(ds, schedule="instant").table.X


def whole_log_device_degree(ds: Dataset) -> np.ndarray:
    """A classic batch-join leak: device degree computed once over the entire log, then joined back."""
    t = ds.txns
    pairs = np.unique(np.stack([t["device"], t["account"]]), axis=1)
    degree = np.bincount(pairs[0], minlength=ds.n_devices).astype(np.float32)
    return np.column_stack([replay(ds).table.X, degree[t["device"]]])


@pytest.fixture(scope="module")
def month(small_dataset: Dataset) -> Dataset:
    return known_at(small_dataset, 24 * DAY)


@pytest.mark.parametrize("cut_day", [9.3, 17.6])
def test_honest_pipeline_is_point_in_time(month, cut_day):
    reports = check_point_in_time(month, int(cut_day * DAY), honest)
    for report in reports:
        assert report.rows_compared > 1000
        assert report.passed, f"{report.variant} leaked through {report.mismatched_features}"


def test_label_leak_is_detected(month):
    reports = check_point_in_time(month, int(15.2 * DAY), instant_labels)
    assert not all(r.passed for r in reports)
    leaked = set().union(*(r.mismatched_features for r in reports))
    assert leaked <= {
        "device_known_fraud",
        "acct_known_fraud",
        "cp_known_fraud",
        "acct_devices_known_fraud",
    } | {n for n in FEATURE_NAMES if "ppr" in n or n.startswith("comm_")}
    assert "device_known_fraud" in leaked


def test_whole_log_aggregate_is_detected(month):
    names = (*FEATURE_NAMES, "device_degree_whole_log")
    reports = check_point_in_time(month, int(15.2 * DAY), whole_log_device_degree, names)
    assert any("device_degree_whole_log" in r.mismatched_features for r in reports)
    assert all(set(r.mismatched_features) <= {"device_degree_whole_log"} for r in reports)


def test_known_at_drops_future_rows_and_late_reports(small_dataset):
    cut = 15 * DAY
    past = known_at(small_dataset, cut)
    assert past.txns["ts"].max() < cut
    reported = past.txns["report_ts"] < NEVER
    assert np.all(past.txns["report_ts"][reported] < cut)
    assert np.all(past.txns["pattern"][~reported] == 0)


def test_scrambled_future_keeps_the_past(small_dataset):
    cut = 15 * DAY
    scrambled = scrambled_future(small_dataset, cut)
    past = small_dataset.txns["ts"] < cut
    for name in ("ts", "account", "device", "merchant", "dst", "amount", "channel"):
        assert np.array_equal(scrambled.txns[name][past], small_dataset.txns[name][past])
    assert not np.array_equal(scrambled.txns["amount"][~past], small_dataset.txns["amount"][~past])
    changed = scrambled.txns["report_ts"] != small_dataset.txns["report_ts"]
    assert np.all(scrambled.txns["report_ts"][changed & past] >= cut)
    assert np.all(small_dataset.txns["pattern"][changed & past] == 0), "fake reports only on genuine rows"


def test_checkpoint_does_not_change_features(small_dataset):
    plain = replay(small_dataset, stop_ts=20 * DAY).table.X
    checked = replay(small_dataset, stop_ts=20 * DAY, checkpoint_ts=12 * DAY)
    assert checked.checkpoint is not None
    assert np.array_equal(plain, checked.table.X)


def test_stopped_replay_is_a_prefix_of_the_full_replay(small_dataset, prepared):
    prefix = replay(small_dataset, stop_ts=18 * DAY + 5 * HOUR).table.X
    assert np.array_equal(prefix, prepared.table.X[: prefix.shape[0]])


def test_snapshot_ignores_edges_after_its_time(small_dataset):
    cut = 14 * DAY
    a = replay(small_dataset, stop_ts=cut + 6 * HOUR).store.snapshot
    b = replay(scrambled_future(small_dataset, cut + HOUR), stop_ts=cut + 6 * HOUR).store.snapshot
    assert a.time == b.time == cut
    assert np.array_equal(a.account_ids, b.account_ids)
    assert np.array_equal(a.node_x, b.node_x)
    assert np.array_equal(a.community, b.community)
