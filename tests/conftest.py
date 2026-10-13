import numpy as np
import pandas as pd
import pytest

from creditrisk.data_loader import BILLS, FEATURES, PAYMENTS, PAY_STATUS


@pytest.fixture
def clients():
    """Deterministic synthetic fixture for tests only; never used for experiment metrics."""
    rng = np.random.default_rng(42)
    n = 160
    frame = pd.DataFrame(0, index=np.arange(n), columns=FEATURES)
    frame["LIMIT_BAL"] = rng.integers(10000, 400000, n)
    frame["SEX"] = rng.integers(1, 3, n)
    frame["EDUCATION"] = rng.integers(1, 5, n)
    frame["MARRIAGE"] = rng.integers(1, 4, n)
    frame["AGE"] = rng.integers(21, 75, n)
    for c in PAY_STATUS:
        frame[c] = rng.integers(-2, 4, n)
    for c in BILLS:
        frame[c] = rng.integers(-1000, 200000, n)
    for c in PAYMENTS:
        frame[c] = rng.integers(0, 20000, n)
    frame.insert(0, "ID", np.arange(1, n + 1))
    frame["default"] = ((frame.PAY_0 > 1) | (rng.random(n) < 0.15)).astype(int)
    return frame
