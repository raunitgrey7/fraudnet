"""Metrics, the cost model, bootstrap intervals and the time split."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fraudnet.evaluation.metrics import (
    CostModel,
    day_block_bootstrap,
    detection_metrics,
    paired_delta_bootstrap,
    precision_at_k_per_day,
    recall_at_fpr,
    threshold_at_fpr,
    top_k_per_day,
)
from fraudnet.evaluation.split import TimeSplit
from fraudnet.sim.config import DAY


def test_recall_at_fpr_on_a_hand_example():
    y = np.array([0] * 100 + [1] * 4)
    score = np.concatenate([np.linspace(0, 0.5, 100), [0.9, 0.8, 0.45, 0.1]])
    assert recall_at_fpr(y, score, 0.01) == pytest.approx(0.5)
    assert recall_at_fpr(y, score, 0.5) == pytest.approx(0.75)


def test_ties_cannot_inflate_recall():
    y = np.array([0] * 100 + [1] * 10)
    score = np.ones(110)
    assert recall_at_fpr(y, score, 0.05) == 0.0
    assert threshold_at_fpr(y, score, 0.05) == 1.0


def test_precision_at_k_per_day_and_budget_mask():
    day = np.array([0, 0, 0, 1, 1, 1])
    y = np.array([1, 0, 0, 0, 1, 1])
    score = np.array([0.9, 0.8, 0.1, 0.7, 0.6, 0.2])
    assert precision_at_k_per_day(day, y, score, 1) == pytest.approx(0.5)
    assert precision_at_k_per_day(day, y, score, 2) == pytest.approx(0.5)
    mask = top_k_per_day(day, score, 2)
    assert mask.tolist() == [1, 1, 0, 1, 1, 0]


def test_cost_model_money_saved():
    cost = CostModel(review_cost=5, false_alert_cost=10, chargeback_fee=15)
    y = np.array([1, 0, 1, 0])
    amount = np.array([100.0, 50.0, 20.0, 30.0])
    alert = np.array([1, 1, 0, 0])
    assert cost.loss_without_model(y, amount) == pytest.approx(150.0)
    assert cost.money_saved(y, amount, alert) == pytest.approx(115 - 10 - 10)


@settings(max_examples=40, deadline=None)
@given(st.lists(st.tuples(st.booleans(), st.floats(1, 500), st.floats(0, 1)), min_size=2, max_size=40))
def test_best_threshold_is_at_least_as_good_as_any_cut(rows):
    y = np.array([int(r[0]) for r in rows])
    amount = np.array([r[1] for r in rows])
    score = np.array([r[2] for r in rows])
    cost = CostModel()
    best = cost.best_threshold(y, amount, score)
    chosen = cost.money_saved(y, amount, (score >= best).astype(int))
    for t in np.unique(score):
        assert chosen >= cost.money_saved(y, amount, (score >= t).astype(int)) - 1e-6
    assert chosen >= 0.0


def test_bootstrap_is_deterministic_and_brackets_the_estimate():
    rng = np.random.default_rng(0)
    day = np.repeat(np.arange(20), 50)
    values = rng.normal(5.0, 1.0, day.size)
    lo, hi = day_block_bootstrap(day, lambda idx: float(values[idx].mean()), 200, seed=3)
    assert (lo, hi) == day_block_bootstrap(day, lambda idx: float(values[idx].mean()), 200, seed=3)
    assert lo < values.mean() < hi


def test_paired_delta_of_identical_models_is_zero():
    rng = np.random.default_rng(1)
    day = np.repeat(np.arange(10), 30)
    y = (rng.random(day.size) < 0.2).astype(int)
    score = rng.random(day.size)
    assert paired_delta_bootstrap(day, y, score, score, 50) == (0.0, 0.0)


def test_detection_metrics_perfect_ranking():
    y = np.array([0] * 300 + [1] * 5)
    score = np.concatenate([np.zeros(300), np.ones(5)])
    m = detection_metrics(y, score, np.zeros(305, dtype=int), (5,))
    assert m["pr_auc"] == pytest.approx(1.0) and m["roc_auc"] == pytest.approx(1.0)
    assert m["recall_at_1pct_fpr"] == 1.0 and m["precision_at_5_per_day"] == 1.0


def test_split_windows_are_disjoint_ordered_and_skip_the_gap():
    split = TimeSplit.for_days(90)
    assert (split.burn_in_day, split.train_end_day, split.val_end_day, split.build_day) == (7, 49, 63, 70)
    ts = np.arange(90) * DAY
    masks = split.masks(ts)
    assert not (masks["train"] & masks["val"]).any() and not (masks["val"] & masks["test"]).any()
    assert not (masks["train"] | masks["val"] | masks["test"])[63:70].any(), "maturity gap is never fitted on"
    assert masks["test"][70:].all()
    no_gap = TimeSplit.for_days(90, gap=False)
    assert no_gap.val_end_day == no_gap.build_day == 70


def test_split_labels_use_build_time_reports(small_dataset):
    split = TimeSplit.for_days(small_dataset.config.n_days)
    labels = split.labels(small_dataset)
    assert labels["train"].sum() < small_dataset.is_fraud.sum()
    assert np.array_equal(labels["test"], small_dataset.is_fraud)
    info = split.describe(small_dataset)
    assert 0.0 < info["train"]["fraud_unlabelled_at_build"] < 1.0
