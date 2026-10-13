"""All learned preprocessing stays inside the estimator pipeline."""

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from creditrisk.data_loader import CATEGORICAL
from creditrisk.feature_engineering import FinancialFeatures


def numeric_columns(frame):
    """Module-level selector remains serializable in trusted model artifacts."""
    return [c for c in frame.columns if c not in CATEGORICAL]


def make_pipeline(estimator, linear: bool = False) -> Pipeline:
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler() if linear else "passthrough"),
        ]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    # Categories are nominal for the linear baseline, including repayment status codes.
    if linear:
        columns = ColumnTransformer(
            [
                ("categorical", categorical, CATEGORICAL),
                ("numeric", numeric, numeric_columns),
            ],
            verbose_feature_names_out=False,
        )
    else:
        columns = ColumnTransformer(
            [("numeric", numeric, make_column_selector(dtype_include="number"))],
            verbose_feature_names_out=False,
        )
    columns.set_output(transform="pandas")
    return Pipeline(
        [("features", FinancialFeatures()), ("preprocess", columns), ("model", estimator)]
    )
