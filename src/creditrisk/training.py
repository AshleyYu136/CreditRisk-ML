"""Full real-data experiment, with test evaluation strictly after selection."""

import json
import logging
import math
import platform
from datetime import datetime, timezone
from importlib.metadata import version

import joblib
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold, train_test_split

from creditrisk.config import ARTIFACTS, load_config
from creditrisk.data_loader import FEATURES, TARGET, download, load_data, sha256, validate
from creditrisk.eda import save_eda, save_evaluation, save_selected
from creditrisk.evaluation import evaluate
from creditrisk.explainability import save_explanations
from creditrisk.feature_engineering import FinancialFeatures
from creditrisk.models import candidates
from creditrisk.monitoring import population_stability
from creditrisk.risk_strategy import select_thresholds, simulate
from creditrisk.scorecard import WOEAnalysis
from creditrisk.sql_analysis import analyze

LOGGER = logging.getLogger(__name__)


def feature_groups(frame):
    """Exact original feature tuples, regardless of labels or customer ID."""
    return frame.groupby(FEATURES, sort=False, dropna=False).ngroup()


def split_data(frame, seed: int):
    groups = feature_groups(frame)
    labels = frame[TARGET].groupby(groups).mean().ge(0.5).astype(int)
    train_groups, holdout_groups = train_test_split(
        labels.index, test_size=0.30, random_state=seed, stratify=labels
    )
    val_groups, test_groups = train_test_split(
        holdout_groups, test_size=0.5, random_state=seed, stratify=labels.loc[holdout_groups]
    )
    return tuple(
        frame.loc[groups.isin(ids)].copy() for ids in [train_groups, val_groups, test_groups]
    )


def serializable_parameters(parameters):
    """Preserve special parameter sentinels as strings; never sanitize metrics."""
    return {
        name: str(value) if isinstance(value, float) and not math.isfinite(value) else value
        for name, value in parameters.items()
    }


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(value, indent=2, allow_nan=False)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(serialized)
    temporary.replace(path)


def train(mode: str | None = None, config_path=None):
    cfg = load_config(config_path)
    if mode:
        cfg["mode"] = mode
    if cfg["mode"] not in {"fast", "comprehensive"}:
        raise ValueError("Unknown training mode")
    full = cfg["mode"] == "comprehensive"
    seed, jobs = cfg["seed"], cfg["n_jobs"]
    for folder in ["models", "metrics", "figures", "evaluation"]:
        (ARTIFACTS / folder).mkdir(parents=True, exist_ok=True)
    # A failed refresh must not expose a mixed, partially overwritten experiment.
    (ARTIFACTS / "metrics/results.json").unlink(missing_ok=True)
    path = download()
    frame = load_data(path)
    quality = validate(frame, expected_rows=30000)
    write_json(ARTIFACTS / "metrics/data_quality.json", quality)
    save_eda(frame, ARTIFACTS / "figures")
    queries = analyze(frame)
    write_json(
        ARTIFACTS / "metrics/sql_summary.json",
        {k: v.head(30).to_dict(orient="records") for k, v in queries.items()},
    )
    aggregation = queries["feature_aggregation"].set_index("ID")
    engineered = FinancialFeatures().fit_transform(frame[FEATURES])
    if not (
        (aggregation.loc[frame.ID, "bill_mean"].to_numpy() - engineered.bill_mean.to_numpy()).round(
            6
        )
        == 0
    ).all():
        raise ValueError("SQL and Pandas bill features differ")
    train_frame, val_frame, test_frame = split_data(frame, seed)
    x, y = train_frame[FEATURES], train_frame[TARGET]
    xv, yv = val_frame[FEATURES], val_frame[TARGET]
    xt, yt = test_frame[FEATURES], test_frame[TARGET]
    cv = StratifiedGroupKFold(n_splits=cfg["cv_folds"], shuffle=True, random_state=seed)
    models, records, validation_predictions = {}, {}, {}
    for name, (pipeline, space) in candidates(seed, jobs, full).items():
        LOGGER.info("Training %s baseline and tuning", name)
        pipeline.fit(x, y)
        baseline = evaluate(yv, pipeline.predict_proba(xv)[:, 1])
        search = RandomizedSearchCV(
            pipeline,
            space,
            n_iter=8 if full else cfg["search_iterations"],
            scoring="roc_auc",
            cv=cv,
            n_jobs=1,
            random_state=seed,
            error_score="raise",
            return_train_score=True,
        )
        search.fit(x, y, groups=feature_groups(train_frame))
        tuned = search.best_estimator_
        p = tuned.predict_proba(xv)[:, 1]
        validation_predictions[name] = p
        models[name] = tuned
        records[name] = {
            "baseline_validation": baseline,
            "validation": evaluate(yv, p),
            "train": evaluate(y, tuned.predict_proba(x)[:, 1]),
            "cv_roc_auc": float(search.best_score_),
            "best_parameters": search.best_params_,
            "model_parameters": serializable_parameters(tuned[-1].get_params()),
            "feature_names": tuned[:-1].get_feature_names_out().tolist(),
        }
        pd.DataFrame(search.cv_results_).to_csv(ARTIFACTS / f"metrics/{name}_cv.csv", index=False)
        joblib.dump(tuned, ARTIFACTS / f"models/{name}.joblib")
    selected = max(records, key=lambda n: records[n]["validation"]["roc_auc"])
    tree_selected = max(
        (n for n in records if n != "logistic_regression"),
        key=lambda n: records[n]["validation"]["roc_auc"],
    )
    thresholds = select_thresholds(
        yv,
        validation_predictions[selected],
        cfg["maximum_validation_bad_rate"],
        cfg["review_threshold"],
    )
    # The choices above are frozen before any test labels are used for evaluation.
    predictions = {}
    for name, model in models.items():
        p = model.predict_proba(xt)[:, 1]
        predictions[name] = p
        records[name]["test"] = evaluate(yt, p)
    results = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": cfg,
        "python": platform.python_version(),
        "versions": {
            p: version(p)
            for p in ["numpy", "pandas", "scikit-learn", "xgboost", "lightgbm", "shap"]
        },
        "dataset_sha256": sha256(path),
        "split_sizes": {"train": len(x), "validation": len(xv), "test": len(xt)},
        "default_rates": {
            "train": float(y.mean()),
            "validation": float(yv.mean()),
            "test": float(yt.mean()),
        },
        "selected_model": selected,
        "selected_tree_model": tree_selected,
        "thresholds": thresholds,
        "validation_strategy": simulate(
            yv, validation_predictions[selected], thresholds["approval"], thresholds["review"]
        ),
        "test_strategy": simulate(
            yt, predictions[selected], thresholds["approval"], thresholds["review"]
        ),
        "models": records,
    }
    joblib.dump(models[selected], ARTIFACTS / "models/selected.joblib")
    joblib.dump(models[tree_selected], ARTIFACTS / "models/selected_tree.joblib")
    # Persist IDs to prove disjoint membership and enable auditing.
    for name, subset in [("train", train_frame), ("validation", val_frame), ("test", test_frame)]:
        subset.to_csv(ARTIFACTS / f"evaluation/{name}.csv", index=False)
    val_output = val_frame[["ID", TARGET]].copy()
    test_output = test_frame[["ID", TARGET]].copy()
    for name in models:
        val_output[name] = validation_predictions[name]
        test_output[name] = predictions[name]
    val_output.to_csv(ARTIFACTS / "evaluation/validation_predictions.csv", index=False)
    test_output.to_csv(ARTIFACTS / "evaluation/test_predictions.csv", index=False)
    comparison = pd.DataFrame(
        [
            {
                "model": n,
                "validation_auc": r["validation"]["roc_auc"],
                "baseline_validation_auc": r["baseline_validation"]["roc_auc"],
                "train_auc": r["train"]["roc_auc"],
                **{
                    k: r["test"][k]
                    for k in [
                        "roc_auc",
                        "average_precision",
                        "ks",
                        "brier",
                        "precision",
                        "recall",
                        "f1",
                    ]
                },
            }
            for n, r in records.items()
        ]
    )
    comparison.to_csv(ARTIFACTS / "metrics/comparison.csv", index=False)
    woe = WOEAnalysis().fit(x.LIMIT_BAL, y)
    woe.table_.to_csv(ARTIFACTS / "metrics/woe_limit.csv")
    write_json(
        ARTIFACTS / "metrics/woe.json",
        {"limit_iv": woe.iv_, "bin_edges": [str(v) for v in woe.edges_]},
    )
    monitor = {
        c: population_stability(x[c], xv[c], categorical=c in ["SEX", "EDUCATION", "MARRIAGE"])
        for c in FEATURES
    }
    monitor["risk_score"] = population_stability(
        models[selected].predict_proba(x)[:, 1], validation_predictions[selected]
    )
    write_json(
        ARTIFACTS / "metrics/monitoring.json",
        {"comparison": "validation holdout, not production", "features": monitor},
    )
    save_evaluation(yt, predictions, records, ARTIFACTS / "figures")
    save_selected(
        yt,
        predictions[selected],
        models[selected],
        records[selected]["test"],
        ARTIFACTS / "figures",
    )
    # Explain validation samples, selected tree chosen solely by validation AUC.
    metadata = save_explanations(models[tree_selected], xv.iloc[:150], ARTIFACTS / "figures")
    write_json(ARTIFACTS / "metrics/shap.json", {"model": tree_selected, **metadata})
    write_json(ARTIFACTS / "metrics/results.json", results)
    LOGGER.info("Selected %s; test AUC %.4f", selected, records[selected]["test"]["roc_auc"])
    return results
