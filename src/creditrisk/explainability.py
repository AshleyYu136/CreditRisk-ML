"""Tree SHAP on the exact transformed model inputs; output scale is recorded."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap


def explain(model, frame):
    transformed = model[:-1].transform(frame)
    names = model[:-1].get_feature_names_out()
    estimator = model[-1]
    # Pin XGBoost 3.0.5: 3.1 vector intercepts are incompatible with SHAP 0.49.1.
    explainer = shap.TreeExplainer(estimator, model_output="raw")
    explanation = explainer(transformed, check_additivity=True)
    if explanation.values.ndim == 3:
        explanation = explanation[:, :, 1]
    explanation.feature_names = list(names)
    explanation.data = transformed.to_numpy()
    scale = (
        "probability"
        if estimator.__class__.__name__ in {"RandomForestClassifier", "DecisionTreeClassifier"}
        else "log_odds"
    )
    total = np.asarray(explanation.base_values) + explanation.values.sum(axis=1)
    predicted = estimator.predict_proba(transformed)[:, 1]
    reconstructed = total if scale == "probability" else 1 / (1 + np.exp(-total))
    if not np.allclose(reconstructed, predicted, atol=1e-5):
        raise ValueError("SHAP additivity does not reconstruct the default probability")
    return explanation, scale


def save_explanations(model, frame, directory):
    explanation, scale = explain(model, frame)
    shap.plots.beeswarm(explanation, max_display=20, show=False)
    plt.title(f"Global SHAP ({scale})")
    plt.tight_layout()
    plt.savefig(directory / "shap_summary.png", dpi=140, bbox_inches="tight")
    plt.close()
    shap.plots.bar(explanation, max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(directory / "shap_importance.png", dpi=140, bbox_inches="tight")
    plt.close()
    shap.plots.waterfall(explanation[0], max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(directory / "shap_waterfall.png", dpi=140, bbox_inches="tight")
    plt.close()
    top = int(np.argmax(np.abs(explanation.values).mean(axis=0)))
    shap.plots.scatter(explanation[:, top], show=False)
    plt.tight_layout()
    plt.savefig(directory / "shap_dependence.png", dpi=140, bbox_inches="tight")
    plt.close()
    return {
        "output_scale": scale,
        "sample_size": len(frame),
        "feature_names": explanation.feature_names,
        "mean_absolute_shap": np.abs(explanation.values).mean(axis=0).tolist(),
    }
