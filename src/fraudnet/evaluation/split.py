"""Strictly time-based train, validation and test windows, with labels as known at model build time.

A random split would put a ring's bust-out in training and its warm-up in test, and would let the model learn
an attacker device from its future. The split is by day: a burn-in period that only warms the feature store,
then train and validation windows, a label-maturity gap, and the test window. The model is built on the
first test day, so training and validation labels are the reports that had arrived by then; fraud reported
later counts as genuine, exactly as it would for a bank retraining on that day. The gap exists because the
most recent week's labels are mostly missing (fraud is reported days to weeks late), and fitting on them
would teach the model that recent fraud is genuine. Test metrics use the full ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fraudnet.sim.config import DAY
from fraudnet.sim.dataset import Dataset


@dataclass(frozen=True)
class TimeSplit:
    """Day boundaries of the burn-in, train, validation, maturity-gap and test windows."""

    burn_in_day: int
    train_end_day: int
    val_end_day: int
    build_day: int
    n_days: int

    @classmethod
    def for_days(cls, n_days: int, gap: bool = True) -> TimeSplit:
        """Default proportions of the horizon: 8 percent burn-in, test from 78 percent.

        With ``gap`` the last 8 percent before the build day is left out of fitting because its labels are
        still immature (for 90 days: train 7 to 49, validation 49 to 63, gap 63 to 70, test 70 to 90).
        Without it, validation runs right up to the build day (train 7 to 56, validation 56 to 70).
        """
        build = round(0.78 * n_days)
        if gap:
            return cls(round(0.08 * n_days), round(0.545 * n_days), round(0.70 * n_days), build, n_days)
        return cls(round(0.08 * n_days), round(0.62 * n_days), build, build, n_days)

    @property
    def build_ts(self) -> int:
        """When the model is trained and deployed: the start of the test window."""
        return self.build_day * DAY

    def masks(self, ts: np.ndarray) -> dict[str, np.ndarray]:
        """Boolean masks of the windows over a timestamp column (the gap is in none of them)."""
        day = ts // DAY
        return {
            "train": (day >= self.burn_in_day) & (day < self.train_end_day),
            "val": (day >= self.train_end_day) & (day < self.val_end_day),
            "test": day >= self.build_day,
        }

    def labels(self, dataset: Dataset) -> dict[str, np.ndarray]:
        """Training and validation labels as known at build time; test labels are the ground truth."""
        as_of = dataset.labels_as_of(self.build_ts)
        return {"train": as_of, "val": as_of, "test": dataset.is_fraud}

    def describe(self, dataset: Dataset) -> dict[str, object]:
        """Window sizes, fraud counts and the share of fraud whose label was not yet visible at build time."""
        masks = self.masks(dataset.txns["ts"])
        truth = dataset.is_fraud.astype(bool)
        as_of = dataset.labels_as_of(self.build_ts).astype(bool)
        out: dict[str, object] = {"build_day": self.build_day, "gap_days": [self.val_end_day, self.build_day]}
        for name, mask in masks.items():
            fraud = int(truth[mask].sum())
            entry = {
                "days": [int(dataset.day[mask].min()), int(dataset.day[mask].max()) + 1]
                if mask.any()
                else [],
                "transactions": int(mask.sum()),
                "fraud": fraud,
                "fraud_rate": round(fraud / max(1, int(mask.sum())), 6),
            }
            if name != "test":
                entry["fraud_unlabelled_at_build"] = round(1.0 - int(as_of[mask].sum()) / max(1, fraud), 4)
            out[name] = entry
        return out
