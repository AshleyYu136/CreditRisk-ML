from streamlit.testing.v1 import AppTest

from creditrisk.config import ROOT


def test_dashboard_without_artifacts(monkeypatch, tmp_path):
    # Isolated blank configuration validates the graceful onboarding state without data download.
    monkeypatch.setattr("creditrisk.config.ARTIFACTS", tmp_path)
    app = AppTest.from_file(str(ROOT / "app/main.py")).run(timeout=30)
    assert not app.exception
    assert any("No trained experiment" in message.value for message in app.info)
    for page in [
        "Model Performance",
        "Individual Risk Prediction",
        "SHAP Explainability",
        "Risk Strategy Simulator",
        "Model Monitoring",
    ]:
        app.sidebar.radio[0].set_value(page).run(timeout=30)
        assert not app.exception
