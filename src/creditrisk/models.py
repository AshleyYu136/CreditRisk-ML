"""Comparable models and modest reproducible CV search spaces."""

from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from creditrisk.preprocessing import make_pipeline


def candidates(seed: int, jobs: int, comprehensive: bool = False) -> dict:
    trees = 300 if comprehensive else 80
    definitions = {
        "logistic_regression": (
            LogisticRegression(max_iter=1500, random_state=seed),
            {"model__C": [0.1, 1, 10], "model__class_weight": [None, "balanced"]},
            True,
        ),
        "decision_tree": (
            DecisionTreeClassifier(random_state=seed, max_depth=5, min_samples_leaf=50),
            {"model__max_depth": [3, 5, 8], "model__min_samples_leaf": [30, 80]},
            False,
        ),
        "random_forest": (
            RandomForestClassifier(
                n_estimators=trees,
                max_depth=10,
                min_samples_leaf=20,
                random_state=seed,
                n_jobs=jobs,
            ),
            {"model__max_depth": [6, 10, None], "model__min_samples_leaf": [10, 30]},
            False,
        ),
        "xgboost": (
            XGBClassifier(
                n_estimators=trees,
                max_depth=3,
                learning_rate=0.08,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=2,
                eval_metric="logloss",
                random_state=seed,
                n_jobs=jobs,
                tree_method="hist",
            ),
            {
                "model__max_depth": [2, 3, 5],
                "model__learning_rate": [0.04, 0.08],
                "model__reg_lambda": [1, 5],
                "model__scale_pos_weight": [1, 3],
            },
            False,
        ),
        "lightgbm": (
            LGBMClassifier(
                n_estimators=trees,
                learning_rate=0.08,
                num_leaves=15,
                max_depth=5,
                min_child_samples=40,
                random_state=seed,
                n_jobs=jobs,
                verbosity=-1,
            ),
            {
                "model__num_leaves": [7, 15, 31],
                "model__max_depth": [3, 5],
                "model__min_child_samples": [30, 80],
                "model__class_weight": [None, "balanced"],
            },
            False,
        ),
    }
    return {
        name: (make_pipeline(model, linear), params)
        for name, (model, params, linear) in definitions.items()
    }
