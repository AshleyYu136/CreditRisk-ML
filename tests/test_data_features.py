import numpy as np
import pytest

from creditrisk.data_loader import BILLS, FEATURES, PAYMENTS, validate
from creditrisk.feature_engineering import FinancialFeatures
from creditrisk.inference import validate_inputs
from creditrisk.models import candidates
from creditrisk.training import split_data


def test_schema_and_bad_values(clients):
    assert validate(clients)["rows"] == 160
    for broken in [
        clients.drop(columns="AGE"),
        clients.assign(SEX=9),
        clients.assign(LIMIT_BAL=0),
        clients.assign(default=1),
    ]:
        with pytest.raises(ValueError):
            validate(broken)
    broken = clients.copy()
    broken.loc[0, "ID"] = broken.loc[1, "ID"]
    with pytest.raises(ValueError):
        validate(broken)


def test_features_safe_ratios_and_no_target(clients):
    x = clients[FEATURES].copy()
    x.loc[0, BILLS] = 0
    x.loc[0, PAYMENTS] = 0
    x.loc[0, "EDUCATION"] = 6
    x.loc[0, "MARRIAGE"] = 0
    result = FinancialFeatures().fit_transform(x)
    assert np.isfinite(result.to_numpy()).all()
    assert result.loc[0, "payment_to_positive_bill"] == 0
    assert result.loc[0, "EDUCATION"] == 4
    assert result.loc[0, "MARRIAGE"] == 3
    assert not {"ID", "default"} & set(result.columns)
    np.testing.assert_allclose(result.bill_mean, x[BILLS].mean(axis=1))
    np.testing.assert_allclose(result.payment_mean, x[PAYMENTS].mean(axis=1))


def test_disjoint_stratified_splits(clients):
    a, b, c = split_data(clients, 42)
    assert (len(a), len(b), len(c)) == (112, 24, 24)
    assert not set(a.ID) & set(b.ID) and not set(a.ID) & set(c.ID) and not set(b.ID) & set(c.ID)
    assert set(a.ID) | set(b.ID) | set(c.ID) == set(clients.ID)
    assert all(s["default"].nunique() == 2 for s in [a, b, c])
    assert a.equals(split_data(clients, 42)[0])


def test_preprocessing_fitted_only_on_training(clients):
    train, val, _ = split_data(clients, 42)
    pipeline = candidates(42, 1)["logistic_regression"][0].fit(train[FEATURES], train["default"])
    scaler = pipeline["preprocess"].named_transformers_["numeric"]["scale"]
    before = scaler.mean_.copy()
    shifted = val[FEATURES].copy()
    shifted.LIMIT_BAL *= 100
    pipeline.predict_proba(shifted)
    np.testing.assert_array_equal(before, scaler.mean_)
    assert len(pipeline[:-1].get_feature_names_out()) == pipeline[-1].n_features_in_


def test_input_contract(clients):
    assert validate_inputs(clients[FEATURES]).shape == (160, 23)
    with pytest.raises(ValueError):
        validate_inputs(clients)
    with pytest.raises(ValueError):
        validate_inputs(clients[FEATURES].assign(PAY_AMT1=-1))
    with pytest.raises(ValueError):
        validate_inputs(clients[FEATURES].assign(AGE=10))


def test_repeated_vectors_never_cross_splits(clients):
    import pandas as pd
    from creditrisk.training import feature_groups

    repeats = clients.iloc[:20].copy()
    repeats.ID += 1000
    repeats["default"] = 1 - repeats["default"]
    combined = pd.concat([clients, repeats], ignore_index=True)
    parts = split_data(combined, 42)
    memberships = [set(map(tuple, part[FEATURES].to_numpy())) for part in parts]
    assert not memberships[0] & memberships[1]
    assert not memberships[0] & memberships[2]
    assert not memberships[1] & memberships[2]
    assert len(set(feature_groups(combined))) == len(clients)


def test_parameter_serialization_keeps_metrics_strict(tmp_path):
    import json
    from creditrisk.training import serializable_parameters, write_json

    path = tmp_path / "result.json"
    write_json(path, {"parameters": serializable_parameters({"missing": float("nan")})})
    assert json.loads(path.read_text())["parameters"]["missing"] == "nan"
    with pytest.raises(ValueError):
        write_json(path, {"metric": float("nan")})
    assert json.loads(path.read_text())["parameters"]["missing"] == "nan"
