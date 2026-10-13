"""Artifact-driven Streamlit dashboard: training is an explicit offline command."""

import json
import logging

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import streamlit as st
from sklearn.metrics import precision_recall_curve, roc_curve

from creditrisk.config import ARTIFACTS, DATA
from creditrisk.data_loader import FEATURES, PAY_STATUS, load_data
from creditrisk.explainability import explain
from creditrisk.inference import load_trusted, predict
from creditrisk.monitoring import population_stability
from creditrisk.risk_strategy import segment, simulate

LOGGER = logging.getLogger(__name__)
st.set_page_config(page_title="CreditRisk · Decision Support", page_icon="📊", layout="wide")
st.title("CreditRisk · Credit Card Default Analytics")
st.caption(
    "UCI historical Taiwan credit card data · Educational decision support · Not a lending policy"
)
PAGES = [
    "Data Overview",
    "Model Performance",
    "Individual Risk Prediction",
    "SHAP Explainability",
    "Risk Strategy Simulator",
    "Model Monitoring",
]
page = st.sidebar.radio("Workspace", PAGES)
st.sidebar.caption("Experiments are trained offline. This dashboard reads saved results.")


@st.cache_data
def read_csv(path, modified):
    return pd.read_csv(path)


def csv(path):
    return read_csv(str(path), path.stat().st_mtime_ns)


@st.cache_resource
def cached_model(path, modified):
    return load_trusted(path, trusted=True)


def model(name="selected"):
    path = ARTIFACTS / f"models/{name}.joblib"
    return cached_model(path, path.stat().st_mtime_ns)


@st.cache_data
def raw_data(path, modified):
    return load_data(path)


def figure(name, caption=None):
    path = ARTIFACTS / f"figures/{name}.png"
    if path.exists():
        st.image(str(path), caption=caption)
    else:
        st.info(f"Figure {name} is unavailable. Run the training pipeline to generate it.")


def render(results):
    selected = results["selected_model"]
    thresholds = results["thresholds"]
    if page == "Data Overview":
        quality = json.loads((ARTIFACTS / "metrics/data_quality.json").read_text())
        cols = st.columns(3)
        cols[0].metric("Customers", f"{quality['rows']:,}")
        cols[1].metric("Next-month default rate", f"{quality['default_rate']:.1%}")
        cols[2].metric("Missing cells", quality["missing"])
        st.write(
            "Default means failure to make the next monthly credit card payment, not loan origination default."
        )
        st.json(quality, expanded=False)
        figure("eda", "Amounts are NT dollars. Segment rates show associations, not causes.")
        figure("target_distributions")
        figure("correlation")
        summaries = json.loads((ARTIFACTS / "metrics/sql_summary.json").read_text())
        for name, rows in summaries.items():
            with st.expander(f"SQL: {name}"):
                st.dataframe(pd.DataFrame(rows), hide_index=True)
        raw = DATA / "raw/default of credit card clients.xls"
        if raw.exists():
            frame = raw_data(raw, raw.stat().st_mtime_ns)
            st.subheader("Historical payment amount distributions")
            st.plotly_chart(
                px.box(
                    frame.melt(
                        value_vars=[f"PAY_AMT{i}" for i in range(1, 7)],
                        var_name="Month",
                        value_name="NT dollars",
                    ),
                    x="Month",
                    y="NT dollars",
                ),
                use_container_width=True,
            )
    elif page == "Model Performance":
        st.write(
            f"Selected **{selected}** by validation ROC-AUC; test scores did not influence selection."
        )
        st.dataframe(csv(ARTIFACTS / "metrics/comparison.csv"), hide_index=True)
        st.caption(
            "Precision, recall, F1 and confusion matrices use probability ≥ 0.5; AUC, AP, KS and Brier do not use this threshold. AP summarizes precision–recall performance."
        )
        predictions = csv(ARTIFACTS / "evaluation/test_predictions.csv")
        cols = st.columns(2)
        roc, pr = go.Figure(), go.Figure()
        for name in results["models"]:
            fpr, tpr, _ = roc_curve(predictions["default"], predictions[name])
            precision, recall, _ = precision_recall_curve(predictions["default"], predictions[name])
            roc.add_scatter(x=fpr, y=tpr, name=name, mode="lines")
            pr.add_scatter(x=recall, y=precision, name=name, mode="lines")
        roc.update_layout(
            title="Held-out ROC curves",
            xaxis_title="False positive rate",
            yaxis_title="True positive rate",
        )
        pr.update_layout(
            title="Held-out precision–recall curves", xaxis_title="Recall", yaxis_title="Precision"
        )
        cols[0].plotly_chart(roc, use_container_width=True)
        cols[1].plotly_chart(pr, use_container_width=True)
        choice = st.selectbox("Inspect model", list(results["models"]))
        metrics = results["models"][choice]["test"]
        st.metric("KS (max TPR − FPR)", f"{metrics['ks']:.3f}")
        st.dataframe(
            pd.DataFrame(
                metrics["confusion_matrix"],
                index=["Observed nondefault", "Observed default"],
                columns=["Predicted nondefault", "Predicted default"],
            )
        )
        calibration = metrics["calibration"]
        chart = go.Figure()
        chart.add_scatter(
            x=calibration["predicted"], y=calibration["observed"], mode="lines+markers", name=choice
        )
        chart.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration")
        chart.update_layout(
            title="Quantile calibration bins",
            xaxis_title="Mean predicted probability",
            yaxis_title="Observed default frequency",
        )
        st.plotly_chart(chart, use_container_width=True)
        figure("selected_model")
        figure("feature_importance")
        st.caption(
            "Training-versus-validation AUC in the table indicates overfitting. Raw probabilities are not guaranteed calibrated; no post-hoc calibration was fitted."
        )
    elif page == "Individual Risk Prediction":
        st.write(
            "Enter the 23 observed features. PAY_0 is the most recent repayment status; PAY_2…PAY_6 cover earlier months. −2/−1/0 are distinct historical codes; positive codes indicate delay in months."
        )
        st.caption(
            "SEX, EDUCATION and MARRIAGE are retained to reproduce the academic benchmark; real decisions require separate fairness review."
        )
        defaults = {c: 0 for c in FEATURES}
        defaults.update(LIMIT_BAL=200000, SEX=2, EDUCATION=2, MARRIAGE=1, AGE=35)
        with st.form("customer"):
            values = {}
            columns = st.columns(3)
            for i, name in enumerate(FEATURES):
                with columns[i % 3]:
                    if name == "SEX":
                        values[name] = st.selectbox(name, [1, 2], index=1)
                    elif name == "EDUCATION":
                        values[name] = st.selectbox(name, list(range(7)), index=2)
                    elif name == "MARRIAGE":
                        values[name] = st.selectbox(name, list(range(4)), index=1)
                    elif name in PAY_STATUS:
                        values[name] = st.selectbox(name, list(range(-2, 10)), index=2)
                    else:
                        minimum = (
                            18
                            if name == "AGE"
                            else 1
                            if name == "LIMIT_BAL"
                            else 0
                            if name.startswith("PAY_AMT")
                            else -1000000
                        )
                        values[name] = st.number_input(
                            name,
                            min_value=minimum,
                            max_value=100 if name == "AGE" else 100000000,
                            value=defaults[name],
                            step=1,
                        )
            submitted = st.form_submit_button("Predict next-month default risk")
        if submitted:
            customer = pd.DataFrame([values])
            p = float(predict(model(), customer)[0])
            tier = segment([p], thresholds["approval"], thresholds["review"])[0]
            st.metric("Predicted default probability", f"{p:.1%}")
            st.write(
                f"Risk tier: **{tier}** · Hypothetical action: **{ {'Low': 'approve', 'Medium': 'manual review', 'High': 'reject'}[tier] }**"
            )
            st.caption(
                "These thresholds were fixed using validation data. This is an illustrative prediction and not financial advice."
            )
            if selected == "logistic_regression":
                fitted = model()
                transformed = fitted[:-1].transform(customer)[0]
                contributions = transformed * fitted[-1].coef_[0]
                factors = pd.DataFrame(
                    {
                        "feature": fitted[:-1].get_feature_names_out(),
                        "log_odds_contribution": contributions,
                    }
                )
                st.dataframe(
                    factors.assign(magnitude=factors.log_odds_contribution.abs())
                    .nlargest(10, "magnitude")
                    .drop(columns="magnitude")
                )
            else:
                explanation, scale = explain(model(), customer)
                factors = pd.DataFrame(
                    {"feature": explanation.feature_names, f"SHAP ({scale})": explanation.values[0]}
                )
                st.dataframe(
                    factors.assign(magnitude=abs(explanation.values[0]))
                    .nlargest(10, "magnitude")
                    .drop(columns="magnitude")
                )
    elif page == "SHAP Explainability":
        metadata = json.loads((ARTIFACTS / "metrics/shap.json").read_text())
        st.write(
            f"Explanations use **{metadata['model']}**, the best validation tree model. Output scale: **{metadata['output_scale']}**."
        )
        st.caption(
            "Global importance averages absolute contributions over 150 validation customers. Local contributions explain model predictions, not why a customer actually defaulted."
        )
        figure("shap_summary")
        figure("shap_importance")
        figure("shap_dependence")
        validation = csv(ARTIFACTS / "evaluation/validation.csv")
        index = st.number_input(
            "Validation customer row", min_value=0, max_value=len(validation) - 1, value=0, step=1
        )
        if st.button("Explain this validation customer"):
            explanation, scale = explain(model("selected_tree"), validation[FEATURES].iloc[[index]])
            probability = predict(model("selected_tree"), validation[FEATURES].iloc[[index]])[0]
            st.metric("Tree model probability", f"{probability:.1%}")
            shap.plots.waterfall(explanation[0], max_display=15, show=False)
            st.pyplot(plt.gcf())
            plt.close()
            st.caption(f"Additive contributions are in {scale}; they are not causal effects.")
        else:
            figure("shap_waterfall", "First validation customer from the saved run")
    elif page == "Risk Strategy Simulator":
        st.info(
            "Explore validation data only. Final test reporting below uses frozen saved thresholds and never follows the sliders."
        )
        approval = st.slider(
            "Approval boundary (risk below this is low)",
            0.01,
            0.49,
            float(thresholds["approval"]),
            0.01,
        )
        review = st.slider(
            "Rejection boundary (risk at or above this is high)",
            0.50,
            1.0,
            float(thresholds["review"]),
            0.01,
        )
        validation = csv(ARTIFACTS / "evaluation/validation_predictions.csv")
        report = simulate(validation["default"], validation[selected], approval, review)
        st.subheader("Validation exploration · historical labels")
        st.json(report)
        st.plotly_chart(
            px.bar(
                x=["Approve", "Review", "Reject"],
                y=[report["approval_rate"], report["review_rate"], report["rejection_rate"]],
                labels={"x": "Hypothetical action", "y": "Fraction of customers"},
            ),
            use_container_width=True,
        )
        st.subheader("Frozen final test strategy")
        st.write(
            f"Approval < {thresholds['approval']:.2f}; manual review < {thresholds['review']:.2f}; rejection otherwise."
        )
        st.json(results["test_strategy"])
        st.caption(
            "No EAD, LGD, interest or costs are observed. This report estimates neither realized profit nor expected credit loss. Validation constraint: maximize approvals with observed default rate ≤10%; fixed review boundary 0.50 is illustrative."
        )
    elif page == "Model Monitoring":
        training = csv(ARTIFACTS / "evaluation/train.csv")
        validation = csv(ARTIFACTS / "evaluation/validation.csv")
        synthetic = st.checkbox("Demonstrate explicitly simulated credit-limit shift", value=False)
        comparison = validation.copy()
        if synthetic:
            comparison["LIMIT_BAL"] *= 1.5
        st.write(
            "Comparison: **simulated +50% credit-limit shift**"
            if synthetic
            else "Comparison: **validation holdout**, not observed production drift"
        )
        reports = {
            c: population_stability(
                training[c], comparison[c], categorical=c in ["SEX", "EDUCATION", "MARRIAGE"]
            )
            for c in FEATURES
        }
        reports["risk_score"] = population_stability(
            predict(model(), training[FEATURES]), predict(model(), comparison[FEATURES])
        )
        table = pd.DataFrame(
            [{"feature": c, "PSI": v["psi"], "flag": v["flag"]} for c, v in reports.items()]
        ).sort_values("PSI", ascending=False)
        st.dataframe(table, hide_index=True)
        choice = st.selectbox("Distribution comparison", list(reports))
        report = reports[choice]
        distributions = pd.DataFrame(
            {
                "bin": report["bins"],
                "Training reference": report["reference"],
                "Comparison": report["comparison"],
            }
        )
        st.plotly_chart(
            px.bar(
                distributions.melt(id_vars="bin", var_name="Sample", value_name="Fraction"),
                x="bin",
                y="Fraction",
                color="Sample",
                barmode="group",
            ),
            use_container_width=True,
        )
        st.caption(
            "Reference quantile bins with infinite tails and an explicit missing bin. Half-count smoothing stabilizes zero frequencies. PSI <0.10 / <0.25 flags are contextual heuristics, not statistical guarantees."
        )


results_path = ARTIFACTS / "metrics/results.json"
if not results_path.exists():
    st.info(
        "No trained experiment is available. Download the UCI data and run the training command first."
    )
    st.code("python scripts/download_data.py\npython scripts/train_models.py --mode fast")
else:
    try:
        render(json.loads(results_path.read_text()))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        LOGGER.exception("Dashboard could not load a valid experiment")
        st.error(
            f"The saved experiment could not be loaded ({type(exc).__name__}). Check your artifact directory and rerun training. Details are in the server log."
        )
