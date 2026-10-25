"""Reason codes from group occlusion: which feature families pushed this score up, in words an analyst reads.

For one transaction, each feature group (velocity, shared device, network risk and so on) is replaced by the
median value of genuine training transactions and the row is re-scored; the drop in score is that group's
contribution. All variants are scored in one batched call, so a reason-coded score costs one model call of
about fourteen rows. Occlusion works for any model with a batch ``predict``, which is why the boosted model
and the GNN share it, and it is honest about interactions in a way a single feature's split gain is not.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from fraudnet.features.schema import FEATURE_INDEX, GROUPS, FeatureGroup, columns_for
from fraudnet.sim.config import CATEGORIES, TRANSFER_CATEGORY

MIN_CONTRIBUTION = 0.005


def _v(row: np.ndarray, name: str) -> float:
    return float(row[FEATURE_INDEX[name]])


def _expm1(row: np.ndarray, name: str) -> float:
    return math.expm1(_v(row, name))


def _category(row: np.ndarray) -> str:
    code = int(_v(row, "merchant_category"))
    return "transfer" if code == TRANSFER_CATEGORY else CATEGORIES[min(code, len(CATEGORIES) - 1)]


DETAILS: dict[str, Callable[[np.ndarray], str]] = {
    "amount": lambda r: (
        f"amount {_expm1(r, 'log_amount'):.2f} is {_v(r, 'amount_z'):.1f} standard deviations from this account's mean"
    ),
    "time": lambda r: (
        f"made at {math.degrees(math.atan2(_v(r, 'hour_sin'), _v(r, 'hour_cos'))) / 15 % 24:.1f} h"
    ),
    "account": lambda r: (
        f"account is {_expm1(r, 'acct_age_days'):.0f} days old with {_expm1(r, 'acct_txn_count'):.0f} earlier transactions"
    ),
    "velocity": lambda r: (
        f"{_v(r, 'acct_cnt_1h'):.0f} transactions in the last hour, {_v(r, 'acct_cnt_24h'):.0f} in the last day"
    ),
    "context": lambda r: (
        f"{_category(r)} payment, first time at this merchant: {'yes' if _v(r, 'merchant_new_for_acct') else 'no'}"
    ),
    "device_account": lambda r: (
        f"device new to this account: {'yes' if _v(r, 'device_new_for_acct') else 'no'}"
    ),
    "flow": lambda r: (
        f"received {_expm1(r, 'acct_inflow_24h'):.0f} in the last day, this payment is {_v(r, 'passthrough_ratio'):.2f} of it"
    ),
    "shared_device": lambda r: (
        f"device used by {_v(r, 'device_accts_total'):.0f} accounts, {_v(r, 'device_accts_24h'):.0f} in the last day"
    ),
    "counterparty": lambda r: (
        f"merchant had {_v(r, 'merchant_small_cnt_1h'):.0f} small charges in the last hour; payee had {_v(r, 'dst_senders_7d'):.0f} senders this week"
    ),
    "risk": lambda r: (
        f"reported fraud on this device: {_expm1(r, 'device_known_fraud'):.0f}, on this account: {_expm1(r, 'acct_known_fraud'):.0f}"
    ),
    "degree": lambda r: (
        f"{_expm1(r, 'acct_deg_merchants'):.0f} merchants and {_expm1(r, 'acct_deg_counterparties'):.0f} payees in 30 days"
    ),
    "pagerank": lambda r: (
        f"fraud proximity score {_v(r, 'acct_ppr_risk'):.2f} (account), {_v(r, 'device_ppr_risk'):.2f} (device)"
    ),
    "community": lambda r: (
        f"cluster of {_expm1(r, 'comm_size'):.0f} accounts, {_v(r, 'comm_new_acct_frac') * 100:.0f} percent opened in the last 30 days"
    ),
}


@dataclass(frozen=True)
class Reason:
    """One reason code with its contribution to the score."""

    code: str
    description: str
    detail: str
    contribution: float

    def to_dict(self) -> dict[str, object]:
        """JSON-ready form."""
        return {
            "code": self.code,
            "description": self.description,
            "detail": self.detail,
            "contribution": round(self.contribution, 4),
        }


class OcclusionExplainer:
    """Turns group contributions into ranked reason codes."""

    def __init__(self, baseline: np.ndarray, groups: tuple[FeatureGroup, ...] = GROUPS) -> None:
        self.baseline = baseline.astype(np.float32)
        self.groups = groups
        self.group_columns = [columns_for([g.name]) for g in groups]

    @classmethod
    def from_training(cls, X: np.ndarray, y: np.ndarray) -> OcclusionExplainer:
        """Baseline = median of genuine training transactions."""
        return cls(np.median(X[y == 0], axis=0))

    def variants(self, row: np.ndarray) -> np.ndarray:
        """The row itself followed by one copy per group with that group set to the baseline."""
        batch = np.repeat(row[None, :].astype(np.float32), 1 + len(self.groups), axis=0)
        for i, cols in enumerate(self.group_columns, start=1):
            batch[i, cols] = self.baseline[cols]
        return batch

    def reasons(self, row: np.ndarray, scores: np.ndarray, top: int = 3) -> list[Reason]:
        """Rank groups by how much replacing them lowers the score (``scores`` from ``variants``)."""
        drops = scores[0] - scores[1:]
        order = np.argsort(-drops, kind="stable")
        out = []
        for i in order[:top]:
            if drops[i] < MIN_CONTRIBUTION:
                break
            group = self.groups[int(i)]
            out.append(
                Reason(group.reason_code, group.description, DETAILS[group.name](row), float(drops[i]))
            )
        return out
