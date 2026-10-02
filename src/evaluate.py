"""Probability-focused evaluation helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score, brier_score_loss, confusion_matrix, log_loss, roc_auc_score


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, p >= 0.5)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
    }


def diagnostics(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> dict:
    """Calibration curve, confusion matrix and predicted-probability histogram."""
    frac_pos, mean_pred = calibration_curve(y, p, n_bins=n_bins, strategy="quantile")
    cm = confusion_matrix(y, (p >= 0.5).astype(int), labels=[0, 1])
    hist, edges = np.histogram(p, bins=20, range=(0, 1))
    return {
        "calibration": {"mean_predicted": mean_pred.tolist(), "fraction_positive": frac_pos.tolist()},
        "confusion_matrix": {"labels": ["B wins", "A wins"], "matrix": cm.tolist()},
        "probability_histogram": {"counts": hist.tolist(), "bin_edges": edges.tolist()},
    }


def date_range(df: pd.DataFrame) -> dict:
    return {
        "start": str(df["event_date"].min().date()),
        "end": str(df["event_date"].max().date()),
        "n_fights": int(len(df)),
    }


def results_table(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows).set_index("model")[["accuracy", "log_loss", "brier", "roc_auc", "n"]].round(4)
