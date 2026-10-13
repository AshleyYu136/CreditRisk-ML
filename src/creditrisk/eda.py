"""Headless figures derived from actual observed data and model predictions."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay

from creditrisk.data_loader import BILLS, PAYMENTS, PAY_STATUS


def save_eda(frame, directory):
    directory.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    frame["default"].value_counts().sort_index().plot.bar(
        ax=axes[0, 0], title="Next-month default (0=no, 1=yes)"
    )
    frame.LIMIT_BAL.plot.hist(bins=40, ax=axes[0, 1], title="Credit limits (NT dollars)")
    frame.AGE.plot.hist(bins=35, ax=axes[0, 2], title="Customer age")
    (frame[PAY_STATUS] > 0).mean().plot.bar(
        ax=axes[1, 0], title="Fraction with positive repayment delay by month"
    )
    frame[PAYMENTS].median().plot.bar(
        ax=axes[1, 1], title="Median historical payments (NT dollars)"
    )
    frame.groupby(pd.cut(frame.AGE, [18, 30, 50, 100], include_lowest=True), observed=True)[
        "default"
    ].mean().plot.bar(ax=axes[1, 2], title="Default rate by age group")
    fig.tight_layout()
    fig.savefig(directory / "eda.png", dpi=140)
    plt.close(fig)
    corr = frame.drop(columns="ID").corr()
    fig, ax = plt.subplots(figsize=(10, 9))
    image = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(corr)), corr.columns, fontsize=7)
    fig.colorbar(image, ax=ax)
    ax.set_title("Observed feature correlations")
    fig.tight_layout()
    fig.savefig(directory / "correlation.png", dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for label in [0, 1]:
        subset = frame[frame["default"] == label]
        axes[0].hist(subset.LIMIT_BAL, bins=40, density=True, alpha=0.45, label=f"default={label}")
        axes[1].hist(
            subset[BILLS].mean(axis=1), bins=60, density=True, alpha=0.45, label=f"default={label}"
        )
    axes[0].set_title("Credit limit by outcome")
    axes[1].set_title("Mean historical bill by outcome")
    for ax in axes:
        ax.legend()
    fig.tight_layout()
    fig.savefig(directory / "target_distributions.png", dpi=140)
    plt.close(fig)


def save_evaluation(y, predictions, results, directory):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for name, p in predictions.items():
        RocCurveDisplay.from_predictions(y, p, name=name, ax=axes[0])
        PrecisionRecallDisplay.from_predictions(y, p, name=name, ax=axes[1])
        calibration = results[name]["test"]["calibration"]
        axes[2].plot(calibration["predicted"], calibration["observed"], marker="o", label=name)
    axes[2].plot([0, 1], [0, 1], "--", color="gray")
    axes[2].set(
        xlabel="Predicted probability",
        ylabel="Observed default rate",
        title="Held-out test calibration",
    )
    axes[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(directory / "evaluation.png", dpi=140)
    plt.close(fig)


def save_selected(y, p, model, metrics, directory):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    cm = np.asarray(metrics["confusion_matrix"])
    axes[0].imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        axes[0].text(j, i, str(v), ha="center")
    axes[0].set(
        xlabel="Predicted default at 0.5", ylabel="Observed default", title="Confusion matrix"
    )
    for label in [0, 1]:
        axes[1].hist(
            p[np.asarray(y) == label], bins=30, alpha=0.5, label=f"default={label}", density=True
        )
    axes[1].legend()
    axes[1].set_title("Held-out risk score distribution")
    fig.tight_layout()
    fig.savefig(directory / "selected_model.png", dpi=140)
    plt.close(fig)
    names = model[:-1].get_feature_names_out()
    estimator = model[-1]
    values = (
        estimator.feature_importances_
        if hasattr(estimator, "feature_importances_")
        else estimator.coef_[0]
    )
    order = np.argsort(np.abs(values))[-20:]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(names[order], values[order])
    ax.set_title("Selected model importance (coefficients for logistic regression)")
    fig.tight_layout()
    fig.savefig(directory / "feature_importance.png", dpi=140)
    plt.close(fig)
