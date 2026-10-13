"""Deterministic features using only six historical months."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from creditrisk.data_loader import BILLS, FEATURES, PAYMENTS, PAY_STATUS


class FinancialFeatures(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X[FEATURES].copy()
        out["EDUCATION"] = out.EDUCATION.replace({0: 4, 5: 4, 6: 4})
        out["MARRIAGE"] = out.MARRIAGE.replace({0: 3})
        out["bill_mean"] = out[BILLS].mean(axis=1)
        out["payment_mean"] = out[PAYMENTS].mean(axis=1)
        out["utilization_proxy"] = out.BILL_AMT1 / out.LIMIT_BAL.clip(lower=1)
        out["payment_to_positive_bill"] = out[PAYMENTS].sum(axis=1) / out[BILLS].clip(lower=0).sum(
            axis=1
        ).clip(lower=1)
        out["max_delay"] = out[PAY_STATUS].clip(lower=0).max(axis=1)
        out["delinquent_months"] = (out[PAY_STATUS] > 0).sum(axis=1)
        out["recent_bill_change"] = out.BILL_AMT1 - out.BILL_AMT6
        return out

    def get_feature_names_out(self, input_features=None):
        return np.asarray(
            [
                *FEATURES,
                "bill_mean",
                "payment_mean",
                "utilization_proxy",
                "payment_to_positive_bill",
                "max_delay",
                "delinquent_months",
                "recent_bill_change",
            ],
            dtype=object,
        )
