"""Validate the feature contract and load only explicitly trusted local artifacts."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from creditrisk.data_loader import CATEGORICAL, FEATURES, PAYMENTS, PAY_STATUS
from creditrisk.evaluation import check_probabilities


def validate_inputs(frame: pd.DataFrame) -> pd.DataFrame:
    if set(frame.columns) != set(FEATURES):
        raise ValueError("Provide exactly the 23 original feature columns, without ID or target")
    out = frame[FEATURES].apply(pd.to_numeric, errors="raise")
    if out.empty or not np.isfinite(out.to_numpy()).all():
        raise ValueError("Provide finite numeric inputs")
    if not out.AGE.between(18, 100).all() or not (out.LIMIT_BAL > 0).all():
        raise ValueError("Age must be 18–100 and credit limit positive")
    allowed = {"SEX": [1, 2], "EDUCATION": range(7), "MARRIAGE": range(4)}
    allowed.update({c: range(-2, 10) for c in PAY_STATUS})
    for c in CATEGORICAL:
        if not out[c].isin(allowed[c]).all():
            raise ValueError(f"Invalid code in {c}")
    if (out[PAYMENTS] < 0).any().any():
        raise ValueError("Payment amounts must be nonnegative")
    return out


def load_trusted(path: Path, *, trusted: bool = False):
    if not trusted:
        raise ValueError("Joblib can execute code; explicitly trust locally trained artifacts only")
    return joblib.load(path)


def predict(model, frame):
    return check_probabilities(model.predict_proba(validate_inputs(frame))[:, 1])
