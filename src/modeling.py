"""Model definitions, calibration wrapper and the symmetric prediction rule.

All predictions (evaluation and the app) go through `predict_proba_symmetric`, so
the reported test metrics describe exactly what the app returns.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import RANDOM_SEED
from src.features import DIFF_FEATURES, MODEL_FEATURES

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:  # pragma: no cover - fallback path
    HAS_XGB = False


# Small, deliberate search spaces (MVP: no large hyperparameter searches).
MODEL_GRIDS: dict[str, list[dict]] = {
    "logistic_regression": [{"C": c} for c in (0.01, 0.1, 1.0)],
    "random_forest": [
        {"n_estimators": 400, "max_depth": d, "min_samples_leaf": leaf, "max_features": "sqrt"}
        for d in (6, 10) for leaf in (20, 50)
    ],
    "xgboost": [
        {"n_estimators": n, "max_depth": d, "learning_rate": 0.03, "subsample": 0.8,
         "colsample_bytree": 0.7, "min_child_weight": 10, "reg_lambda": 5.0}
        for d in (2, 3) for n in (200, 400)
    ],
}


def make_model(kind: str, params: dict):
    if kind == "logistic_regression":
        return Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=params["C"], max_iter=2000, random_state=RANDOM_SEED)),
        ])
    if kind == "random_forest":
        return RandomForestClassifier(**params, n_jobs=-1, random_state=RANDOM_SEED)
    if kind == "xgboost":
        if HAS_XGB:
            return XGBClassifier(**params, eval_metric="logloss", n_jobs=4, random_state=RANDOM_SEED)
        return HistGradientBoostingClassifier(
            max_depth=params["max_depth"], learning_rate=params["learning_rate"],
            max_iter=params["n_estimators"], l2_regularization=params["reg_lambda"],
            random_state=RANDOM_SEED,
        )
    raise ValueError(kind)


class Calibrator:
    """Maps a raw probability to a calibrated one (identity, Platt/sigmoid or isotonic)."""

    def __init__(self, method: str = "none"):
        self.method = method
        self._model = None

    def fit(self, p: np.ndarray, y: np.ndarray) -> "Calibrator":
        if self.method == "sigmoid":
            z = _logit(p).reshape(-1, 1)
            self._model = LogisticRegression(C=1e6, max_iter=1000).fit(z, y)
        elif self.method == "isotonic":
            self._model = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip").fit(p, y)
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        if self.method == "sigmoid":
            return self._model.predict_proba(_logit(p).reshape(-1, 1))[:, 1]
        if self.method == "isotonic":
            return self._model.predict(p)
        return p


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class FightModel:
    """Base classifier + calibrator, predicting P(A beats B) symmetrically."""

    def __init__(self, kind: str, params: dict, estimator, calibrator: Calibrator | None = None):
        self.kind = kind
        self.params = params
        self.estimator = estimator
        self.calibrator = calibrator or Calibrator("none")

    def raw_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Uncalibrated, un-symmetrised P(A wins) for rows as given."""
        return self.estimator.predict_proba(X[MODEL_FEATURES])[:, 1]

    def oriented_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Calibrated but not symmetrised."""
        return self.calibrator.transform(self.raw_proba(X))

    def predict_proba_symmetric(self, X: pd.DataFrame) -> np.ndarray:
        """P(A wins) = mean of p(A,B) and 1 - p(B,A), after calibration.

        Guarantees P(A beats B) + P(B beats A) = 1 exactly.
        """
        return 0.5 * (self.oriented_proba(X) + 1 - self.oriented_proba(swap(X)))


def swap(X: pd.DataFrame) -> pd.DataFrame:
    Xs = X.copy()
    Xs[DIFF_FEATURES] = -Xs[DIFF_FEATURES]
    return Xs
