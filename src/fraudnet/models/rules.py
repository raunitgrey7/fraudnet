"""Rules baseline: the hand-written checks a fraud team runs before it has a model.

Each rule reads the same point-in-time features as the learned models, carries a weight that reflects how
strongly an analyst would trust it, and has a reason code. The score is the sum of fired weights plus a small
amount-based tie-breaker, which mirrors how rule queues are worked in practice (largest amounts first among
equal rule scores) and gives the baseline a full ranking for PR-AUC. Weights and cut-offs were set from
domain reasoning before any model was trained, not fitted to the data.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from fraudnet.features.schema import FEATURE_INDEX
from fraudnet.sim.config import HIGH_RISK_CATEGORIES, TRANSFER_CATEGORY

RULE_ALERT_SCORE = 3.0


def _col(X: np.ndarray, name: str) -> np.ndarray:
    return X[:, FEATURE_INDEX[name]]


def _high_risk(X: np.ndarray) -> np.ndarray:
    return np.isin(_col(X, "merchant_category"), list(HIGH_RISK_CATEGORIES))


def _hour(X: np.ndarray) -> np.ndarray:
    angle = np.arctan2(_col(X, "hour_sin"), _col(X, "hour_cos"))
    return np.mod(angle * 24.0 / (2.0 * math.pi), 24.0)


@dataclass(frozen=True)
class Rule:
    """One rule: a vectorised predicate over the feature matrix, a weight and a reason code."""

    code: str
    description: str
    weight: float
    predicate: Callable[[np.ndarray], np.ndarray]


RULES: tuple[Rule, ...] = (
    Rule("VELOCITY_BURST", "three or more transactions on the account in the last hour", 2.0,
         lambda X: _col(X, "acct_cnt_1h") >= 3),
    Rule("HIGH_AMOUNT_NEW_DEVICE", "amount 2.5 standard deviations above the account mean on a new device", 3.0,
         lambda X: (_col(X, "amount_z") >= 2.5) & (_col(X, "device_new_for_acct") > 0)),
    Rule("DEVICE_FANOUT", "device used by three or more accounts in the last 24 hours", 3.0,
         lambda X: _col(X, "device_accts_24h") >= 3),
    Rule("SHARED_DEVICE", "device used by five or more accounts", 2.0,
         lambda X: _col(X, "device_accts_total") >= 5),
    Rule("CARD_TESTING_BURST", "charge under 5 at a merchant with eight or more small charges this hour", 3.0,
         lambda X: (_col(X, "log_amount") < math.log1p(5.0)) & (_col(X, "merchant_small_cnt_1h") >= 8)),
    Rule("PASS_THROUGH", "spends or forwards 70 percent of money received within six hours", 3.0,
         lambda X: (_col(X, "passthrough_ratio") >= 0.7)
         & (_col(X, "log_secs_since_inflow") < math.log1p(6 * 3600))
         & ((_col(X, "merchant_category") == TRANSFER_CATEGORY) | _high_risk(X))),
    Rule("NEW_PAYEE_LARGE", "transfer of 500 or more to a first-time payee", 1.0,
         lambda X: (_col(X, "payee_new") > 0) & (_col(X, "log_amount") >= math.log1p(500.0))),
    Rule("NIGHT_HIGH_RISK", "electronics, gift card or crypto purchase between midnight and 5 am", 1.0,
         lambda X: (_hour(X) < 5.0) & _high_risk(X)),
    Rule("NEW_ACCOUNT_HIGH_RISK", "account younger than 14 days buying in a high-risk category", 1.0,
         lambda X: (_col(X, "acct_age_days") < math.log1p(14.0)) & _high_risk(X)),
    Rule("NEW_DEVICE_HIGH_RISK", "new device buying in a high-risk category", 1.0,
         lambda X: (_col(X, "device_new_for_acct") > 0) & _high_risk(X)),
    Rule("KNOWN_FRAUD_DEVICE", "device linked to previously reported fraud", 4.0,
         lambda X: _col(X, "device_known_fraud") > 0),
)  # fmt: skip


class RulesModel:
    """Weighted rule sum with reason codes; needs no training."""

    name = "rules"

    def fired(self, X: np.ndarray) -> np.ndarray:
        """Boolean matrix (rows by rules) of which rules fire."""
        return np.column_stack([rule.predicate(X) for rule in RULES])

    def score(self, X: np.ndarray) -> np.ndarray:
        """Rule score plus an amount tie-breaker below the smallest rule weight."""
        weights = np.array([rule.weight for rule in RULES])
        return self.fired(X).astype(float) @ weights + 0.01 * _col(X, "log_amount")

    def reasons(self, row: np.ndarray) -> list[dict[str, object]]:
        """Fired rules of one transaction, strongest first."""
        hits = self.fired(row[None, :])[0]
        fired = [rule for rule, hit in zip(RULES, hits, strict=True) if hit]
        fired.sort(key=lambda r: -r.weight)
        return [{"code": r.code, "description": r.description, "contribution": r.weight} for r in fired]
