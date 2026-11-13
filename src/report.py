import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import shutil
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
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

from src.baselines import MODELS, SETUPS

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
MODELS_DIR = ROOT / "models"
MODEL_DIRS = {
    "cvd": "cardiovascular",
    "diabetes_full": "diabetes",
    "diabetes_risk_only": "diabetes",
}
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


class KerasProba:
    _estimator_type = "classifier"

    def __init__(self, model):
        self.model = model
        self.classes_ = np.array([0, 1])

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        proba = self.model.predict(X, verbose=0).ravel()
        return np.column_stack([1 - proba, proba])


def build_summary():
    baselines = pd.read_csv(RESULTS_DIR / "baselines.csv")
    mlp = pd.read_csv(RESULTS_DIR / "mlp.csv")
    summary = pd.concat([baselines, mlp], ignore_index=True).reindex(columns=SUMMARY_COLUMNS)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(RESULTS_DIR / "summary.csv", index=False)
    print(summary.to_string(index=False))
    return summary


def plot_roc(setup, probas, y_test):
    fig, ax = plt.subplots(figsize=(6, 6))
    for name in ("logreg", "rf", "mlp"):
        fpr, tpr, _ = roc_curve(y_test, probas[name])
        auc = roc_auc_score(y_test, probas[name])
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"{setup} - ROC")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"roc_{setup}.png", dpi=150)
    plt.close(fig)


def plot_calibration(setup, probas, y_test):
    n_bins = 5 if setup == "cvd" else 10
    fig, ax = plt.subplots(figsize=(6, 6))
    for name in ("logreg", "rf", "mlp"):
        fraction_positive, mean_predicted = calibration_curve(
            y_test, probas[name], n_bins=n_bins, strategy="quantile"
        )
        ax.plot(mean_predicted, fraction_positive, marker="o", label=name)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(f"{setup} - calibration")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"calibration_{setup}.png", dpi=150)
    plt.close(fig)


def plot_confusion(setup, final_name, proba, y_test):
    y_pred = (proba >= 0.5).astype(int)
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred)).plot(
        ax=ax, colorbar=False
    )
    ax.set_title(f"{setup} - confusion ({final_name})")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"confusion_{setup}.png", dpi=150)
    plt.close(fig)


def plot_importance(setup, final_name, top):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top.index[::-1], top.values[::-1])
    ax.set_xlabel("Permutation importance (ROC-AUC drop)")
    ax.set_title(f"{setup} - top 10 features ({final_name})")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"importance_{setup}.png", dpi=150)
    plt.close(fig)


def main():
    summary = build_summary()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    for setup, loader in SETUPS.items():
        X_train, X_test, y_train, y_test, _ = loader()
        model_dir = MODELS_DIR / MODEL_DIRS[setup]
        model_dir.mkdir(parents=True, exist_ok=True)

        fitted = {}
        probas = {}
        for name, factory in MODELS.items():
            model = factory()
            model.fit(X_train, y_train)
            fitted[name] = model
            probas[name] = model.predict_proba(X_test)[:, 1]

        mlp_path = model_dir / f"mlp_{setup}.keras"
        mlp = tf.keras.models.load_model(mlp_path)
        probas["mlp"] = mlp.predict(X_test, verbose=0).ravel()

        plot_roc(setup, probas, y_test)
        plot_calibration(setup, probas, y_test)

        subset = summary[summary["setup"] == setup]
        best_index = subset["cv_roc_auc_mean"].idxmax()
        final_name = subset.loc[best_index, "model"]
        final_cv = subset.loc[best_index, "cv_roc_auc_mean"]
        print(f"\n[{setup}] final model: {final_name} (cv_roc_auc_mean={final_cv:.4f})")

        if final_name == "mlp":
            shutil.copy2(mlp_path, model_dir / f"final_{setup}.keras")
            estimator = KerasProba(mlp)
        else:
            joblib.dump(fitted[final_name], model_dir / f"final_{setup}.joblib")
            estimator = fitted[final_name]

        plot_confusion(setup, final_name, probas[final_name], y_test)

        result = permutation_importance(
            estimator, X_test, y_test, scoring="roc_auc", n_repeats=20, random_state=SEED
        )
        importances = pd.Series(result.importances_mean, index=X_test.columns).sort_values(
            ascending=False
        )
        top = importances.head(10)
        plot_importance(setup, final_name, top)
        print(f"[{setup}] top 10 permutation importances ({final_name}):")
        for feature in top.index:
            pos = list(X_test.columns).index(feature)
            print(
                f"  {feature}: {result.importances_mean[pos]:.4f} "
                f"+/- {result.importances_std[pos]:.4f}"
            )


if __name__ == "__main__":
    main()
