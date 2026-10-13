"""Educational retrospective decisions, never a validated lending policy."""

import numpy as np
from sklearn.metrics import confusion_matrix

from creditrisk.evaluation import check_probabilities


def segment(p, approval: float = 0.15, review: float = 0.4):
    if not 0 <= approval < review <= 1:
        raise ValueError("Require 0 <= approval < review <= 1")
    p = check_probabilities(p)
    return np.where(p < approval, "Low", np.where(p < review, "Medium", "High"))


def simulate(y, p, approval: float = 0.15, review: float = 0.4) -> dict:
    y = np.asarray(y)
    tiers = segment(p, approval, review)
    accepted, manual, rejected = tiers == "Low", tiers == "Medium", tiers == "High"
    return {
        "approval_rate": float(accepted.mean()),
        "review_rate": float(manual.mean()),
        "rejection_rate": float(rejected.mean()),
        "approved_default_rate": float(y[accepted].mean()) if accepted.any() else None,
        "default_capture_review_or_reject": float(y[~accepted].sum() / y.sum())
        if y.sum()
        else None,
        "reviewed_default_rate": float(y[manual].mean()) if manual.any() else None,
        "rejected_default_rate": float(y[rejected].mean()) if rejected.any() else None,
        "default_capture_review": float(y[manual].sum() / y.sum()) if y.sum() else None,
        "default_capture_rejected": float(y[rejected].sum() / y.sum()) if y.sum() else None,
        "confusion_nonapproval": confusion_matrix(y, ~accepted, labels=[0, 1]).tolist(),
    }


def select_thresholds(y, p, maximum_bad_rate: float = 0.10, review: float = 0.5) -> dict:
    # Validation-only constraint: maximize approvals with observed bad rate <= 10%.
    if not 0 <= maximum_bad_rate <= 1 or not 0.05 < review <= 1:
        raise ValueError("Invalid strategy configuration")
    choices = []
    for approval in np.linspace(0.05, min(0.35, review - 0.01), 31):
        report = simulate(y, p, float(approval), review)
        if (
            report["approved_default_rate"] is not None
            and report["approved_default_rate"] <= maximum_bad_rate
        ):
            choices.append((report["approval_rate"], float(approval)))
    approval = max(choices)[1] if choices else 0.05
    return {
        "approval": approval,
        "review": review,
        "maximum_validation_bad_rate": maximum_bad_rate,
        "constraint_met": bool(choices),
    }
