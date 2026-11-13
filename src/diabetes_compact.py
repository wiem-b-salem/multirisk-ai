from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.baselines import MODELS, evaluate
from src.diabetes.preprocess import load_and_split as load_diabetes

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = ROOT / "results" / "diabetes_compact.csv"
SUMMARY_PATH = ROOT / "results" / "summary.csv"
MODELS_DIR = ROOT / "models" / "diabetes"


def main():
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    X_train, X_test, y_train, y_test, scaler = load_diabetes("compact")

    rows = []
    fitted = {}
    for name, factory in MODELS.items():
        model = factory()
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        rows.append(
            {
                "setup": "diabetes_compact",
                "model": name,
                "cv_roc_auc_mean": scores.mean(),
                "cv_roc_auc_std": scores.std(),
                **evaluate(y_test, proba),
            }
        )
        fitted[name] = model

    results = pd.DataFrame(rows)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_PATH, index=False)
    print(results.to_string(index=False))

    summary = pd.read_csv(SUMMARY_PATH)
    full = summary[summary["setup"] == "diabetes_full"]
    best_full = full.loc[full["cv_roc_auc_mean"].idxmax()]
    comparison = pd.concat(
        [
            pd.DataFrame(
                [
                    {
                        "setup": "diabetes_full",
                        "model": best_full["model"],
                        "cv_roc_auc_mean": best_full["cv_roc_auc_mean"],
                        "test_roc_auc": best_full["test_roc_auc"],
                    }
                ]
            ),
            results[["setup", "model", "cv_roc_auc_mean", "test_roc_auc"]],
        ],
        ignore_index=True,
    )
    print("\nComparison:")
    print(comparison.to_string(index=False))

    best_index = results["cv_roc_auc_mean"].idxmax()
    best_name = results.loc[best_index, "model"]
    print(f"\nChosen compact model: {best_name}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted[best_name], MODELS_DIR / "final_diabetes_compact.joblib")
    joblib.dump(scaler, MODELS_DIR / "scaler_diabetes_compact.joblib")


if __name__ == "__main__":
    main()
