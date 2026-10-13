"""PSI with reference-only bins, explicit missing bins and smoothed frequencies."""

import numpy as np
import pandas as pd


def population_stability(reference, comparison, categorical: bool = False, bins: int = 10) -> dict:
    ref, cmp = pd.Series(reference), pd.Series(comparison)
    if len(ref) == 0 or len(cmp) == 0:
        raise ValueError("PSI requires nonempty samples")
    if categorical:
        # Union retains categories new to comparison rather than discarding their mass.
        labels = sorted(set(ref.dropna().astype(str)) | set(cmp.dropna().astype(str))) + [
            "<MISSING>"
        ]

        def counts(series):
            return (
                series.astype("string")
                .fillna("<MISSING>")
                .value_counts()
                .reindex(labels, fill_value=0)
                .to_numpy()
            )
    else:
        finite = pd.to_numeric(ref.dropna(), errors="raise").to_numpy()
        if not np.isfinite(finite).all():
            raise ValueError("Nonfinite reference values")
        internal = (
            np.unique(np.quantile(finite, np.linspace(0, 1, bins + 1)[1:-1])) if len(finite) else []
        )
        edges = np.r_[-np.inf, internal, np.inf]
        labels = [f"({a:g}, {b:g}]" for a, b in zip(edges[:-1], edges[1:])] + ["<MISSING>"]

        def counts(series):
            vals = pd.to_numeric(series.dropna(), errors="raise").to_numpy()
            if not np.isfinite(vals).all():
                raise ValueError("Nonfinite comparison values")
            indexes = np.searchsorted(edges[1:-1], vals, side="left")
            return np.r_[np.bincount(indexes, minlength=len(edges) - 1), series.isna().sum()]

    r, c = counts(ref), counts(cmp)
    r = (r + 0.5) / (r.sum() + 0.5 * len(r))
    c = (c + 0.5) / (c.sum() + 0.5 * len(c))
    contributions = (c - r) * np.log(c / r)
    psi = float(contributions.sum())
    return {
        "psi": psi,
        "flag": "stable" if psi < 0.1 else "watch" if psi < 0.25 else "investigate",
        "bins": labels,
        "reference": r.tolist(),
        "comparison": c.tolist(),
        "contributions": contributions.tolist(),
    }
