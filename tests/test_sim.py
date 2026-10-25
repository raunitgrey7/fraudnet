"""Simulator: determinism, invariants of the log, planted pattern structure and the dataset cache."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from fraudnet.sim import Dataset, Pattern, SimConfig, simulate
from fraudnet.sim.config import DAY, NEVER, TRANSFER_CATEGORY, Channel
from fraudnet.sim.dataset import cache_path
from fraudnet.sim.simulate import load_or_simulate


def test_same_seed_same_bytes():
    assert simulate(SimConfig.small(seed=11)).checksum() == simulate(SimConfig.small(seed=11)).checksum()


def test_different_seed_different_world():
    assert simulate(SimConfig.small(seed=11)).checksum() != simulate(SimConfig.small(seed=12)).checksum()


def test_full_world_fraud_rate_is_under_one_percent():
    summary = simulate(SimConfig()).summary()
    assert 0.003 < summary["fraud_rate"] < 0.01
    assert summary["transactions"] > 300_000


def _check_invariants(ds: Dataset) -> None:
    t = ds.txns
    assert np.all(np.diff(t["ts"]) >= 0), "log must be sorted by time"
    assert t["ts"].min() >= 0 and t["ts"].max() < ds.config.n_days * DAY
    assert np.all(np.isfinite(t["amount"])) and np.all(t["amount"] > 0)
    transfer = t["dst"] >= 0
    assert np.all(t["merchant"][transfer] == -1) and np.all(t["merchant"][~transfer] >= 0)
    assert np.all((t["channel"] == Channel.P2P) == transfer)
    assert np.all(t["dst"][transfer] != t["account"][transfer])
    signup = ds.accounts["signup_ts"]
    assert np.all(signup[t["account"]] <= t["ts"]), "no account transacts before it signs up"
    assert np.all(t["device"] < ds.n_devices) and np.all(t["merchant"] < ds.merchants["category"].size)


def test_log_invariants(small_dataset):
    _check_invariants(small_dataset)


@settings(max_examples=8, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(min_value=0, max_value=10_000))
def test_log_invariants_hold_for_any_seed(seed):
    _check_invariants(simulate(SimConfig.small(seed=seed)))


def test_every_pattern_is_planted(small_dataset):
    summary = small_dataset.summary()
    for name, stats in summary["patterns"].items():
        assert stats["transactions"] > 0, name
        assert stats["attacks"] > 0, name


def test_reports_arrive_after_the_fraud_and_never_for_genuine(small_dataset):
    t = small_dataset.txns
    fraud = t["pattern"] > Pattern.LEGIT
    reported = fraud & (t["report_ts"] < NEVER)
    assert np.all(t["report_ts"][reported] > t["ts"][reported])
    assert np.all(t["report_ts"][~fraud] == NEVER)
    assert 0.0 < reported.sum() / fraud.sum() < 1.0, "some fraud is never reported"


def test_labels_as_of_hide_late_reports(small_dataset):
    t = small_dataset.txns
    cut = 20 * DAY
    as_of = small_dataset.labels_as_of(cut)
    late = (t["pattern"] > 0) & (t["report_ts"] >= cut)
    assert late.any()
    assert not as_of[late].any()
    assert as_of.sum() < small_dataset.is_fraud.sum()


def test_ring_accounts_share_devices_and_are_new(small_dataset):
    t = small_dataset.txns
    ring = t["pattern"] == Pattern.RING
    for attack in np.unique(t["attack_id"][ring]):
        rows = ring & (t["attack_id"] == attack)
        accounts = np.unique(t["account"][rows])
        per_device = [
            np.unique(t["account"][rows & (t["device"] == d)]).size for d in np.unique(t["device"][rows])
        ]
        assert max(per_device) >= 2, "ring members share at least one device"
        assert np.all(small_dataset.accounts["is_ring"][accounts] == 1)
        assert 2 <= accounts.size <= SimConfig.small().ring_size_max
        assert np.all(t["ts"][rows] - small_dataset.accounts["signup_ts"][t["account"][rows]] < 40 * DAY)


def test_card_testing_charges_are_small_and_fan_out_over_cards(small_dataset):
    t = small_dataset.txns
    tests = (t["pattern"] == Pattern.CARD_TESTING) & (t["amount"] <= 10.0)
    assert tests.sum() > 0
    for attack in np.unique(t["attack_id"][tests]):
        rows = tests & (t["attack_id"] == attack)
        assert np.unique(t["account"][rows]).size >= 5, "one attack tests many cards"


def test_mule_money_arrives_before_it_leaves(small_dataset):
    t = small_dataset.txns
    mule = t["pattern"] == Pattern.MULE
    checked = 0
    for attack in np.unique(t["attack_id"][mule])[:20]:
        rows = np.flatnonzero(mule & (t["attack_id"] == attack))
        received = {}
        for i in rows:
            if t["dst"][i] >= 0:
                received.setdefault(int(t["dst"][i]), int(t["ts"][i]))
        for i in rows:
            sender = int(t["account"][i])
            if sender in received:
                assert t["ts"][i] >= received[sender]
                checked += 1
    assert checked > 0


def test_takeover_devices_are_reused_across_victims(small_dataset):
    t = small_dataset.txns
    ato = t["pattern"] == Pattern.ATO
    victims_per_device = [
        np.unique(t["account"][ato & (t["device"] == d)]).size for d in np.unique(t["device"][ato])
    ]
    assert max(victims_per_device) >= 2


def test_genuine_look_alikes_exist(small_dataset):
    t = small_dataset.txns
    legit = t["pattern"] == Pattern.LEGIT
    big_transfers = legit & (t["dst"] >= 0) & (t["amount"] > 500)
    tiny = legit & (t["amount"] < 5)
    assert big_transfers.sum() > 0, "large genuine transfers overlap with mule hops"
    assert tiny.sum() > 0, "small genuine charges overlap with card testing"
    shared = [
        np.unique(t["account"][legit & (t["device"] == d)]).size for d in np.unique(t["device"][legit])[:2000]
    ]
    assert max(shared) >= 2, "households and workplaces share devices"


def test_transfer_category_code_is_outside_merchant_categories(small_dataset):
    assert small_dataset.merchants["category"].max() < TRANSFER_CATEGORY


def test_save_load_roundtrip_verifies_checksum(tmp_path, small_dataset):
    path = tmp_path / "world.npz"
    small_dataset.save(path)
    loaded = Dataset.load(path)
    assert loaded.checksum() == small_dataset.checksum()
    assert loaded.config == small_dataset.config


def test_corrupted_archive_is_rejected(tmp_path, small_dataset):
    path = tmp_path / "world.npz"
    small_dataset.save(path)
    with np.load(path) as archive:
        arrays = {k: archive[k] for k in archive.files}
    arrays["txn_amount"] = arrays["txn_amount"] * 1.01
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="checksum"):
        Dataset.load(path)


def test_cache_is_keyed_by_config(tmp_path):
    a, b = SimConfig.small(seed=1), SimConfig.small(seed=2)
    assert cache_path(a, tmp_path) != cache_path(b, tmp_path)
    first = load_or_simulate(a, tmp_path)
    assert cache_path(a, tmp_path).exists()
    assert load_or_simulate(a, tmp_path).checksum() == first.checksum()
