"""Training-only quantile bins and smoothed WOE/IV; association is not causation."""

import numpy as np
import pandas as pd


class WOEAnalysis:
    def __init__(self, bins: int = 10, smoothing: float = 0.5):
        self.bins = bins
        self.smoothing = smoothing

    def fit(self, x, y):
        x, y = pd.Series(x).reset_index(drop=True), pd.Series(y).reset_index(drop=True)
        if not set(y.unique()) == {0, 1}:
            raise ValueError("WOE requires both binary classes")
        inner = np.unique(x.dropna().quantile(np.linspace(0, 1, self.bins + 1)[1:-1]))
        self.edges_ = np.r_[-np.inf, inner, np.inf]
        binned = pd.cut(x, self.edges_).astype("string").fillna("<MISSING>")
        table = (
            pd.DataFrame({"bin": binned, "bad": y})
            .groupby("bin", observed=True)
            .bad.agg(["count", "sum"])
        )
        table["good"] = table["count"] - table["sum"]
        good = (table.good + self.smoothing) / (table.good.sum() + self.smoothing * len(table))
        bad = (table["sum"] + self.smoothing) / (table["sum"].sum() + self.smoothing * len(table))
        table["bad_rate"] = table["sum"] / table["count"]
        table["woe"] = np.log(good / bad)
        table["iv_contribution"] = (good - bad) * table.woe
        self.table_ = table
        self.iv_ = float(table.iv_contribution.sum())
        self.mapping_ = table.woe.to_dict()
        return self

    def transform(self, x):
        return (
            pd.cut(pd.Series(x), self.edges_)
            .astype("string")
            .fillna("<MISSING>")
            .map(self.mapping_)
            .fillna(0)
            .to_numpy(dtype=float)
        )
