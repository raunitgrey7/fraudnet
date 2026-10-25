"""Detection metrics, the cost model and day-block bootstrap intervals.

Fraud teams do not read ROC-AUC; they read how much fraud is caught at a tolerable false positive rate, how
many of the alerts an analyst opens each day are real, and how much money the model saves after paying for
those reviews. All of those are here, computed the same way for every model.

Recall at a false positive rate uses the threshold that flags at most that share of genuine transactions
(scores strictly above the threshold are flagged), so ties can only make a model look worse, never better.
Confidence intervals resample whole days, because fraud arrives in attacks and transactions within a day
are not independent.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def threshold_at_fpr(y: np.ndarray, score: np.ndarray, fpr: float) -> float:
    """Smallest threshold whose strict exceedances include at most ``fpr`` of the negatives."""
    negatives = np.sort(score[y == 0])[::-1]
    k = int(np.floor(fpr * negatives.size))
    if negatives.size == 0:
        return float("-inf")
    return float(negatives[min(k, negatives.size - 1)])


def recall_at_fpr(y: np.ndarray, score: np.ndarray, fpr: float) -> float:
    """Share of fraud flagged when at most ``fpr`` of genuine transactions are flagged."""
    if y.sum() == 0:
        return float("nan")
    return float((score[y == 1] > threshold_at_fpr(y, score, fpr)).mean())


def precision_at_k_per_day(day: np.ndarray, y: np.ndarray, score: np.ndarray, k: int) -> float:
    """Mean over days of the fraud share among the ``k`` highest-scored transactions of that day."""
    values = []
    for d in np.unique(day):
        mask = day == d
        top = np.argsort(-score[mask], kind="stable")[:k]
        values.append(float(y[mask][top].mean()))
    return float(np.mean(values)) if values else float("nan")


@dataclass(frozen=True)
class CostModel:
    """The stated cost model.

    * every alert costs an analyst review (``review_cost``),
    * an alert on a genuine payment also costs customer friction (``false_alert_cost``): the payment is held
      or declined and some customers leave,
    * a missed fraud costs its amount plus a chargeback fee (``chargeback_fee``),
    * a caught fraud costs only its review.
    """

    review_cost: float = 5.0
    false_alert_cost: float = 10.0
    chargeback_fee: float = 15.0

    def loss_without_model(self, y: np.ndarray, amount: np.ndarray) -> float:
        """What fraud costs when nothing is reviewed."""
        return float(((amount + self.chargeback_fee) * y).sum())

    def money_saved(self, y: np.ndarray, amount: np.ndarray, alert: np.ndarray) -> float:
        """Loss avoided by reviewing ``alert`` transactions, net of review and false-alert costs."""
        caught = ((amount + self.chargeback_fee) * (y * alert)).sum()
        false_alerts = ((1 - y) * alert).sum()
        return float(caught - self.review_cost * alert.sum() - self.false_alert_cost * false_alerts)

    def best_threshold(self, y: np.ndarray, amount: np.ndarray, score: np.ndarray) -> float:
        """Threshold that maximises money saved on the given (validation) data; flag ``score >= threshold``."""
        order = np.argsort(-score, kind="stable")
        gain = (amount[order] + self.chargeback_fee) * y[order] - self.false_alert_cost * (1 - y[order])
        value = np.cumsum(gain - self.review_cost)
        sorted_scores = score[order]
        boundary = np.r_[sorted_scores[1:] != sorted_scores[:-1], True]
        candidates = np.flatnonzero(boundary)
        if candidates.size == 0:
            return float("inf")
        best = candidates[int(np.argmax(value[candidates]))]
        return float(sorted_scores[best]) if value[best] > 0 else float("inf")


def top_k_per_day(day: np.ndarray, score: np.ndarray, k: int) -> np.ndarray:
    """Alert mask when analysts can review only the ``k`` highest-scored transactions of each day."""
    alert = np.zeros(score.size, dtype=np.int64)
    for d in np.unique(day):
        idx = np.flatnonzero(day == d)
        alert[idx[np.argsort(-score[idx], kind="stable")[:k]]] = 1
    return alert


def day_block_bootstrap(
    day: np.ndarray,
    metric: Callable[[np.ndarray], float],
    n_boot: int = 200,
    seed: int = 0,
) -> tuple[float, float]:
    """95 percent interval of ``metric(indices)`` when whole days are resampled with replacement."""
    rng = np.random.default_rng(seed)
    days = np.unique(day)
    groups = [np.flatnonzero(day == d) for d in days]
    stats = []
    for _ in range(n_boot):
        picks = rng.integers(0, days.size, days.size)
        idx = np.concatenate([groups[p] for p in picks])
        value = metric(idx)
        if np.isfinite(value):
            stats.append(value)
    if not stats:
        return float("nan"), float("nan")
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)


def paired_delta_bootstrap(
    day: np.ndarray,
    y: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    n_boot: int = 200,
    seed: int = 0,
) -> tuple[float, float]:
    """95 percent interval of AP(a) minus AP(b) with the same resampled days for both models."""

    def delta(idx: np.ndarray) -> float:
        if not y[idx].any():
            return float("nan")
        return float(
            average_precision_score(y[idx], score_a[idx]) - average_precision_score(y[idx], score_b[idx])
        )

    return day_block_bootstrap(day, delta, n_boot, seed)


def detection_metrics(
    y: np.ndarray, score: np.ndarray, day: np.ndarray, k_values: tuple[int, ...] = (10, 50)
) -> dict[str, float]:
    """PR-AUC, ROC-AUC, recall at 0.5 and 1 percent FPR and precision at top-k alerts per day."""
    out = {
        "pr_auc": float(average_precision_score(y, score)) if y.any() else float("nan"),
        "roc_auc": float(roc_auc_score(y, score)) if 0 < y.sum() < y.size else float("nan"),
        "recall_at_0.5pct_fpr": recall_at_fpr(y, score, 0.005),
        "recall_at_1pct_fpr": recall_at_fpr(y, score, 0.01),
    }
    for k in k_values:
        out[f"precision_at_{k}_per_day"] = precision_at_k_per_day(day, y, score, k)
    return out
