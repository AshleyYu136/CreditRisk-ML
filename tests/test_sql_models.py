import numpy as np
import pytest
import joblib

from creditrisk.data_loader import BILLS, FEATURES
from creditrisk.explainability import explain
from creditrisk.inference import load_trusted, predict
from creditrisk.models import candidates
from creditrisk.sql_analysis import analyze


def test_sql_matches_pandas(clients):
    result = analyze(clients)
    assert int(result["data_quality"].iloc[0]["rows"]) == len(clients)
    sql = result["default_analysis"].set_index("latest_status")
    equivalent = clients.groupby("PAY_0")["default"].mean()
    np.testing.assert_allclose(sql.loc[equivalent.index, "default_rate"], equivalent)
    agg = result["feature_aggregation"].set_index("ID")
    np.testing.assert_allclose(agg.loc[clients.ID, "bill_mean"], clients[BILLS].mean(axis=1))
    np.testing.assert_allclose(
        agg.loc[clients.ID, "mean_limit_by_outcome"],
        clients.groupby("default").LIMIT_BAL.transform("mean"),
    )
    assert sql.risk_rank.min() == 1


@pytest.mark.parametrize("name", list(candidates(42, 1)))
def test_all_model_prediction_and_artifact(name, clients, tmp_path):
    pipeline = candidates(42, 1)[name][0]
    pipeline.fit(clients[FEATURES].iloc[:120], clients["default"].iloc[:120])
    p = predict(pipeline, clients[FEATURES].iloc[120:])
    assert p.shape == (40,) and ((p >= 0) & (p <= 1)).all()
    path = tmp_path / "model.joblib"
    joblib.dump(pipeline, path)
    with pytest.raises(ValueError):
        load_trusted(path)
    np.testing.assert_allclose(
        p, predict(load_trusted(path, trusted=True), clients[FEATURES].iloc[120:])
    )


@pytest.mark.parametrize("name", ["decision_tree", "random_forest", "lightgbm", "xgboost"])
def test_shap_compatibility_and_additivity(name, clients):
    pipeline = candidates(42, 1)[name][0].fit(clients[FEATURES], clients["default"])
    explanation, scale = explain(pipeline, clients[FEATURES].iloc[:3])
    assert explanation.values.shape == (3, len(pipeline[:-1].get_feature_names_out()))
    assert scale in {"probability", "log_odds"}
    total = explanation.base_values + explanation.values.sum(axis=1)
    p = total if scale == "probability" else 1 / (1 + np.exp(-total))
    np.testing.assert_allclose(
        p, pipeline.predict_proba(clients[FEATURES].iloc[:3])[:, 1], atol=1e-5
    )
