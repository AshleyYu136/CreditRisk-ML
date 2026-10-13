import numpy as np
import pytest

from creditrisk.evaluation import check_probabilities, evaluate, ks_statistic
from creditrisk.monitoring import population_stability
from creditrisk.risk_strategy import segment, select_thresholds, simulate
from creditrisk.scorecard import WOEAnalysis


def test_ks_direction_and_evaluation():
    y = [0, 0, 1, 1]
    p = [0.1, 0.2, 0.8, 0.9]
    assert ks_statistic(y, p) == 1
    assert ks_statistic(y, p[::-1]) == 0
    assert ks_statistic(y, [0.5] * 4) == 0
    assert evaluate(y, p)["confusion_matrix"] == [[2, 0], [0, 2]]
    assert evaluate(y, p)["average_precision"] == 1
    with pytest.raises(ValueError):
        ks_statistic([1, 1], [0.1, 0.2])


@pytest.mark.parametrize("p", [[-1], [1.01], [np.nan], [[0.5]], [np.inf]])
def test_invalid_probability(p):
    with pytest.raises(ValueError):
        check_probabilities(p)


def test_strategy_boundaries_and_no_approvals():
    assert segment([0.149, 0.15, 0.399, 0.4], 0.15, 0.4).tolist() == [
        "Low",
        "Medium",
        "Medium",
        "High",
    ]
    with pytest.raises(ValueError):
        segment([0.1], 0.4, 0.4)
    report = simulate([0, 1], [0.1, 0.9], 0.15, 0.4)
    assert report["approval_rate"] == 0.5 and report["default_capture_review_or_reject"] == 1
    assert simulate([0, 1], [0.5, 0.9], 0.15, 0.4)["approved_default_rate"] is None
    selected = select_thresholds([0, 0, 1, 1], [0.02, 0.08, 0.5, 0.9])
    assert selected["constraint_met"]


def test_numeric_psi_missing_and_shift():
    ref = np.r_[np.arange(100), np.nan]
    assert population_stability(ref, ref)["psi"] == 0
    shifted = population_stability(ref, np.arange(200, 300))
    assert shifted["psi"] > 0.25
    assert np.isclose(sum(shifted["reference"]), 1)
    assert np.isclose(sum(shifted["comparison"]), 1)
    assert np.isfinite(population_stability([1, 1, np.nan], [1, 2, np.nan])["psi"])
    assert np.isfinite(population_stability([np.nan], [1])["psi"])
    with pytest.raises(ValueError):
        population_stability([], [1])


def test_categorical_psi_new_category():
    result = population_stability(["a", "a", "b", None], ["c", "c", None], True)
    assert result["psi"] > 0 and "c" in result["bins"]
    assert sum(result["comparison"]) == pytest.approx(1)


def test_woe_training_only_and_zero_count_bins():
    woe = WOEAnalysis(bins=3).fit([1, 2, 3, 4, 5, 6], [0, 0, 0, 1, 1, 1])
    assert np.isfinite(woe.table_.woe).all() and woe.iv_ > 0
    original = woe.edges_.copy()
    assert np.isfinite(woe.transform([-100, 100, np.nan])).all()
    np.testing.assert_array_equal(original, woe.edges_)
