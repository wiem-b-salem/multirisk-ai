from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.cardiovascular.preprocess import load_and_split as load_cvd
from src.diabetes.preprocess import load_and_split as load_diabetes

SEED = 42
RESULTS_PATH = Path(__file__).resolve().parents[1] / "results" / "baselines.csv"

SETUPS = {
    "cvd": lambda: load_cvd(),
    "diabetes_full": lambda: load_diabetes("full"),
    "diabetes_risk_only": lambda: load_diabetes("risk_only"),
}

MODELS = {
    "logreg": lambda: LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=SEED
    ),
    "rf": lambda: RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=SEED
    ),
}


def evaluate(y_true, y_proba):
    y_pred = (y_proba >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "test_roc_auc": roc_auc_score(y_true, y_proba),
        "test_pr_auc": average_precision_score(y_true, y_proba),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "specificity": tn / (tn + fp),
        "f1": f1_score(y_true, y_pred),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def main():
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows = []
    importances = {}
    for setup, loader in SETUPS.items():
        X_train, X_test, y_train, y_test, _ = loader()
        for name, factory in MODELS.items():
            model = factory()
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
            model.fit(X_train, y_train)
            y_proba = model.predict_proba(X_test)[:, 1]
            rows.append(
                {
                    "setup": setup,
                    "model": name,
                    "cv_roc_auc_mean": scores.mean(),
                    "cv_roc_auc_std": scores.std(),
                    **evaluate(y_test, y_proba),
                }
            )
            if name == "rf":
                importances[setup] = (
                    pd.Series(model.feature_importances_, index=X_train.columns)
                    .sort_values(ascending=False)
                    .head(8)
                )

    results = pd.DataFrame(rows)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_PATH, index=False)
    print(results.to_string(index=False))

    for setup, top in importances.items():
        print(f"\n=== {setup} top 8 RF importances ===")
        print(top.round(4).to_string())


if __name__ == "__main__":
    main()
