"""Recompute metrics from frozen saved test predictions; no selection or tuning."""

import json

import numpy as np
import pandas as pd

from creditrisk.config import ARTIFACTS
from creditrisk.evaluation import evaluate

if __name__ == "__main__":
    results = json.loads((ARTIFACTS / "metrics/results.json").read_text())
    predictions = pd.read_csv(ARTIFACTS / "evaluation/test_predictions.csv")
    for name, record in results["models"].items():
        actual = evaluate(predictions["default"], predictions[name])
        for key in ["roc_auc", "ks", "brier", "average_precision", "precision", "recall", "f1"]:
            if not np.isclose(actual[key], record["test"][key]):
                raise ValueError(f"Saved metrics differ: {name}/{key}")
        print(
            name, {k: round(actual[k], 4) for k in ["roc_auc", "ks", "average_precision", "brier"]}
        )
