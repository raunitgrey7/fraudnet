"""A point-in-time correctness checker that any feature pipeline can be run through.

The check is the definition of "no leakage" made executable: the features of a transaction must depend only
on what the bank knew when it happened. So the checker builds the dataset as it was known at a cut time
(transactions before the cut, and only the fraud reports that had arrived by then), runs the pipeline on it,
and requires the features of every transaction before the cut to be bit-identical to the full run. A second
variant keeps the past and scrambles the future (redrawn amounts, devices and merchants, plus fake reports
for past transactions arriving after the cut). A pipeline that peeks at the future, or uses labels before
they were reported, fails one of the two.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np

from fraudnet.features.schema import FEATURE_NAMES
from fraudnet.sim.config import NEVER, Pattern
from fraudnet.sim.dataset import Dataset

Pipeline = Callable[[Dataset], np.ndarray]


@dataclass(frozen=True)
class LeakageReport:
    """Outcome of one comparison between a full run and a run on information available at the cut."""

    variant: str
    cut_ts: int
    rows_compared: int
    values_compared: int
    mismatched_values: int
    mismatched_features: tuple[str, ...]

    @property
    def passed(self) -> bool:
        """True when every compared value is identical."""
        return self.mismatched_values == 0


def known_at(dataset: Dataset, cut_ts: int) -> Dataset:
    """The dataset as known at ``cut_ts``: earlier transactions, and only reports that arrived before it."""
    keep = dataset.txns["ts"] < cut_ts
    txns = {k: v[keep].copy() for k, v in dataset.txns.items()}
    late = txns["report_ts"] >= cut_ts
    txns["pattern"][late] = Pattern.LEGIT
    txns["attack_id"][late] = -1
    txns["report_ts"][late] = NEVER
    return replace(dataset, txns=txns)


def scrambled_future(dataset: Dataset, cut_ts: int, seed: int = 0) -> Dataset:
    """Same past, different future: redraw everything after the cut and add late reports for the past.

    The fake reports go on genuine past transactions only: moving the report of a real fraud that was already
    known before the cut would change the past, and the check would then fail for a correct pipeline.
    """
    rng = np.random.default_rng(seed)
    txns = {k: v.copy() for k, v in dataset.txns.items()}
    future = np.flatnonzero(txns["ts"] >= cut_ts)
    purchases = future[txns["merchant"][future] >= 0]
    txns["amount"][future] = np.round(
        rng.permutation(txns["amount"][future]) * rng.uniform(0.5, 2.0, future.size), 2
    )
    txns["device"][future] = rng.integers(0, dataset.n_devices, future.size)
    txns["merchant"][purchases] = rng.integers(0, dataset.merchants["category"].size, purchases.size)
    past_genuine = np.flatnonzero((txns["ts"] < cut_ts) & (txns["pattern"] == Pattern.LEGIT))
    fake = rng.choice(past_genuine, size=min(past_genuine.size, 200), replace=False)
    txns["pattern"][fake] = Pattern.RING
    txns["report_ts"][fake] = cut_ts + rng.integers(0, 5 * 86_400, fake.size)
    return replace(dataset, txns=txns)


def _compare(
    variant: str, cut_ts: int, full: np.ndarray, other: np.ndarray, names: tuple[str, ...]
) -> LeakageReport:
    rows = int(min(full.shape[0], other.shape[0]))
    a, b = full[:rows], other[:rows]
    diff = ~((a == b) | (np.isnan(a) & np.isnan(b)))
    bad = np.flatnonzero(diff.any(axis=0))
    return LeakageReport(
        variant=variant,
        cut_ts=cut_ts,
        rows_compared=rows,
        values_compared=int(diff.size),
        mismatched_values=int(diff.sum()),
        mismatched_features=tuple(names[j] if j < len(names) else f"col{j}" for j in bad),
    )


def check_point_in_time(
    dataset: Dataset, cut_ts: int, pipeline: Pipeline, names: tuple[str, ...] = FEATURE_NAMES
) -> list[LeakageReport]:
    """Run both variants of the check for one cut time."""
    full = pipeline(dataset)
    n_past = int(np.searchsorted(dataset.txns["ts"], cut_ts, side="left"))
    prefix = pipeline(known_at(dataset, cut_ts))
    scrambled = pipeline(scrambled_future(dataset, cut_ts))[:n_past]
    return [
        _compare("prefix_replay", cut_ts, full[:n_past], prefix, names),
        _compare("scrambled_future", cut_ts, full[:n_past], scrambled, names),
    ]
