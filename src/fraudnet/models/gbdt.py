"""Gradient boosting on tabular and graph features (scikit-learn ``HistGradientBoostingClassifier``).

Histogram boosting is the standard choice for fraud tables: it handles mixed scales and categorical codes
natively, needs no feature scaling and trains in seconds on a laptop. Two details matter here:

* Negatives are down-sampled for training speed and given matching sample weights, so predicted
  probabilities stay calibrated in expectation.
* The number of trees is chosen on the time-based validation window (average precision every twenty
  trees), not on a random hold-out that would mix future and past. Validation labels are the reports known
  at build time, so the chosen size reflects what a bank could actually measure.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits

from fraudnet.features.schema import CATEGORICAL_FEATURES, FEATURE_NAMES


@dataclass(frozen=True)
class GBDTConfig:
    """Hyperparameters; learning rate and leaf count won the validation-only sweep in ``fraudnet.tune``."""

    learning_rate: float = 0.05
    max_leaf_nodes: int = 31
    min_samples_leaf: int = 40
    l2_regularization: float = 1.0
    max_iter: int = 400
    check_every: int = 20
    negative_rate: float = 0.2
    threads: int = 2
    seed: int = 0


class GBDTModel:
    """Boosted trees over a chosen subset of feature columns."""

    def __init__(self, columns: list[int], config: GBDTConfig | None = None, name: str = "gbdt") -> None:
        self.columns = list(columns)
        self.config = config or GBDTConfig()
        self.name = name
        self.clf: HistGradientBoostingClassifier | None = None
        self.n_iter = 0
        self.val_curve: list[tuple[int, float]] = []

    @property
    def feature_names(self) -> list[str]:
        """Names of the columns this model reads."""
        return [FEATURE_NAMES[c] for c in self.columns]

    def _new(self, max_iter: int) -> HistGradientBoostingClassifier:
        mask = np.array([FEATURE_NAMES[c] in CATEGORICAL_FEATURES for c in self.columns])
        cfg = self.config
        return HistGradientBoostingClassifier(
            learning_rate=cfg.learning_rate,
            max_leaf_nodes=cfg.max_leaf_nodes,
            min_samples_leaf=cfg.min_samples_leaf,
            l2_regularization=cfg.l2_regularization,
            max_iter=max_iter,
            early_stopping=False,
            categorical_features=mask if mask.any() else None,
            random_state=cfg.seed,
        )

    def _sample(self, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """All positives and a seeded share of negatives, with weights that undo the sampling."""
        rng = np.random.default_rng(self.config.seed)
        keep = (y == 1) | (rng.random(y.size) < self.config.negative_rate)
        idx = np.flatnonzero(keep)
        weight = np.where(y[idx] == 1, 1.0, 1.0 / self.config.negative_rate)
        return idx, weight

    def fit(
        self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, refit: bool = False
    ) -> GBDTModel:
        """Grow ``max_iter`` trees, pick the tree count with the best validation average precision.

        Validation scores come from staged predictions, so the selection costs one pass over the trees.
        With ``refit`` the model is rebuilt with exactly that many trees (boosting here is deterministic, so
        the trees are the same), which makes single-row serving fast.
        """
        cfg = self.config
        idx, weight = self._sample(y)
        Xs, ys, Xv = X[idx][:, self.columns], y[idx], X_val[:, self.columns]
        with threadpool_limits(cfg.threads):
            clf = self._new(cfg.max_iter).fit(Xs, ys, sample_weight=weight)
            best_iter, best_ap = 0, -1.0
            for i, proba in enumerate(clf.staged_predict_proba(Xv), start=1):
                if i % cfg.check_every == 0 or i == clf.n_iter_:
                    ap = float(average_precision_score(y_val, proba[:, 1]))
                    self.val_curve.append((i, ap))
                    if ap > best_ap + 1e-4:
                        best_iter, best_ap = i, ap
            if refit and best_iter < clf.n_iter_:
                clf = self._new(best_iter).fit(Xs, ys, sample_weight=weight)
        self.clf, self.n_iter = clf, best_iter
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Fraud probability for each row of a full-width feature matrix, using the selected tree count."""
        if self.clf is None:
            raise RuntimeError("model is not fitted")
        Xc = X[:, self.columns]
        with threadpool_limits(self.config.threads):
            if self.clf.n_iter_ == self.n_iter:
                return self.clf.predict_proba(Xc)[:, 1]
            for i, proba in enumerate(self.clf.staged_predict_proba(Xc), start=1):
                if i == self.n_iter:
                    return proba[:, 1]
        raise RuntimeError("selected tree count exceeds the fitted trees")

    def predict_fast(self, X: np.ndarray) -> np.ndarray:
        """Serving path: no thread-pool context switch; requires a refitted model."""
        if self.clf is None or self.clf.n_iter_ != self.n_iter:
            return self.predict(X)
        return self.clf.predict_proba(X[:, self.columns])[:, 1]
