"""Validate every dashboard page and inference using a completed local experiment."""

import json

from streamlit.testing.v1 import AppTest

from creditrisk.config import ARTIFACTS, ROOT


def check(app, context):
    if app.exception or app.error:
        raise RuntimeError(f"Dashboard failed at {context}: {app.exception} {app.error}")


def main():
    results = json.loads((ARTIFACTS / "metrics/results.json").read_text())
    app = AppTest.from_file(str(ROOT / "app/main.py")).run(timeout=60)
    check(app, "Data Overview")
    for page in [
        "Model Performance",
        "Individual Risk Prediction",
        "SHAP Explainability",
        "Risk Strategy Simulator",
        "Model Monitoring",
    ]:
        app.sidebar.radio[0].set_value(page).run(timeout=60)
        check(app, page)
        if page == "Individual Risk Prediction":
            app.button[0].click().run(timeout=60)
            check(app, "individual prediction submit")
            probability = float(app.metric[0].value.rstrip("%")) / 100
            assert 0 <= probability <= 1
        elif page == "SHAP Explainability":
            app.button[0].click().run(timeout=60)
            check(app, "local SHAP waterfall")
            assert app.metric
        elif page == "Risk Strategy Simulator":
            # Sliders alter validation output; the last JSON remains the frozen test report.
            app.slider[0].set_value(0.20).run(timeout=60)
            check(app, "validation threshold exploration")
            assert json.loads(app.json[-1].value) == results["test_strategy"]
        elif page == "Model Monitoring":
            app.checkbox[0].set_value(True).run(timeout=60)
            check(app, "explicit synthetic drift")
    print(
        "PASS: all six pages, real model prediction, local SHAP, frozen test policy and simulated PSI"
    )


if __name__ == "__main__":
    main()
