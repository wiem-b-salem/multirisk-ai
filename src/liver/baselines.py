from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.baselines import MODELS, evaluate
from src.liver.preprocess import load_and_split as load_liver

SEED = 42
ROOT = Path(__file__).resolve().parents[2]
RESULTS_PATH = ROOT / "results" / "liver_baselines.csv"

SETUPS = {
    "liver_full": lambda: load_liver("full"),
    "liver_no_lab": lambda: load_liver("no_lab"),
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
