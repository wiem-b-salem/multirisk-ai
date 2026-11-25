import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

from src.baselines import MODELS
from src.liver.preprocess import load_and_split as load_liver

SEED = 42
SETUP = "liver_full"
ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
MODELS_DIR = ROOT / "models" / "liver"
SUMMARY_COLUMNS = [
    "setup",
    "model",
    "cv_roc_auc_mean",
    "cv_roc_auc_std",
    "test_roc_auc",
    "test_pr_auc",
    "precision",
    "recall",
    "specificity",
    "f1",
    "tn",
    "fp",
    "fn",
    "tp",
    "test_roc_auc_seed_mean",
    "test_roc_auc_seed_std",
]


def build_summary():
    baselines = pd.read_csv(RESULTS_DIR / "liver_baselines.csv")
    mlp = pd.read_csv(RESULTS_DIR / "liver_mlp.csv")
    summary = pd.concat([baselines, mlp], ignore_index=True).reindex(columns=SUMMARY_COLUMNS)
    summary.to_csv(RESULTS_DIR / "liver_summary.csv", index=False)
    print(summary.to_string(index=False))
    return summary


def plot_roc(probas, y_test):
    fig, ax = plt.subplots(figsize=(6, 6))
    for name in ("logreg", "rf", "mlp"):
        fpr, tpr, _ = roc_curve(y_test, probas[name])
        auc = roc_auc_score(y_test, probas[name])
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"{SETUP} - ROC")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"liver_roc_{SETUP}.png", dpi=150)
    plt.close(fig)


def plot_calibration(probas, y_test):
    fig, ax = plt.subplots(figsize=(6, 6))
    for name in ("logreg", "rf", "mlp"):
        fraction_positive, mean_predicted = calibration_curve(
            y_test, probas[name], n_bins=10, strategy="quantile"
        )
        ax.plot(mean_predicted, fraction_positive, marker="o", label=name)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(f"{SETUP} - calibration")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"liver_calibration_{SETUP}.png", dpi=150)
    plt.close(fig)


def plot_confusion(final_name, proba, y_test):
    y_pred = (proba >= 0.5).astype(int)
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred)).plot(
        ax=ax, colorbar=False
    )
    ax.set_title(f"{SETUP} - confusion ({final_name})")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"liver_confusion_{SETUP}.png", dpi=150)
    plt.close(fig)


def plot_importance(final_name, top):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top.index[::-1], top.values[::-1])
    ax.set_xlabel("Permutation importance (ROC-AUC drop)")
    ax.set_title(f"{SETUP} - top 5 features ({final_name})")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"liver_importance_{SETUP}.png", dpi=150)
    plt.close(fig)


def main():
    summary = build_summary()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    X_train, X_test, y_train, y_test, scaler = load_liver("full")

    fitted = {}
    probas = {}
    for name, factory in MODELS.items():
        model = factory()
        model.fit(X_train, y_train)
        fitted[name] = model
        probas[name] = model.predict_proba(X_test)[:, 1]

    mlp = tf.keras.models.load_model(MODELS_DIR / "mlp_liver_full.keras")
    probas["mlp"] = mlp.predict(X_test, verbose=0).ravel()

    plot_roc(probas, y_test)
    plot_calibration(probas, y_test)

    subset = summary[(summary["setup"] == SETUP) & (summary["model"].isin(MODELS))]
    best_index = subset["cv_roc_auc_mean"].idxmax()
    final_name = subset.loc[best_index, "model"]
    final_cv = subset.loc[best_index, "cv_roc_auc_mean"]
    if final_cv < 0.65:
        raise RuntimeError(
            f"Best full-set cv_roc_auc_mean {final_cv:.4f} is below 0.65; stopping."
        )
    print(f"\n[{SETUP}] final model: {final_name} (cv_roc_auc_mean={final_cv:.4f})")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted[final_name], MODELS_DIR / f"final_{SETUP}.joblib")
    joblib.dump(scaler, MODELS_DIR / f"scaler_{SETUP}.joblib")

    plot_confusion(final_name, probas[final_name], y_test)

    result = permutation_importance(
        fitted[final_name], X_test, y_test, scoring="roc_auc", n_repeats=20, random_state=SEED
    )
    importances = pd.Series(result.importances_mean, index=X_test.columns).sort_values(
        ascending=False
    )
    top = importances.head(5)
    plot_importance(final_name, top)
    print(f"[{SETUP}] top 5 permutation importances ({final_name}):")
    for feature in top.index:
        pos = list(X_test.columns).index(feature)
        print(
            f"  {feature}: {result.importances_mean[pos]:.4f} "
            f"+/- {result.importances_std[pos]:.4f}"
        )


if __name__ == "__main__":
    main()
