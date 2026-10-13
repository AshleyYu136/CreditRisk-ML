"""Metrics with high scores consistently indicating default risk."""

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def check_probabilities(p):
    p = np.asarray(p, dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Expected finite one-dimensional probabilities in [0,1]")
    return p


def ks_statistic(y, p) -> float:
    p = check_probabilities(p)
    if len(np.unique(y)) != 2:
        raise ValueError("KS needs both classes")
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def evaluate(y, p, threshold: float = 0.5) -> dict:
    p = check_probabilities(p)
    predicted = p >= threshold
    fraction, mean = calibration_curve(y, p, n_bins=10, strategy="quantile")
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "ks": ks_statistic(y, p),
        "brier": float(brier_score_loss(y, p)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "threshold": threshold,
        "confusion_matrix": confusion_matrix(y, predicted, labels=[0, 1]).tolist(),
        "calibration": {"observed": fraction.tolist(), "predicted": mean.tolist()},
    }
