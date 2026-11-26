import sys

sys.dont_write_bytecode = True

from pathlib import Path

import joblib
import pandas as pd
from sklearn.inspection import permutation_importance

from src.liver.preprocess import load_and_split

ROOT = Path(__file__).resolve().parents[2]
SEED = 42
SETUP = "liver_full"
MODEL_NAME = "rf"
MODEL_FILE = ROOT / "models" / "liver" / "final_liver_full.joblib"
OUTPUT_FILE = ROOT / "results" / "liver_importance_output.txt"


def main():
    X_train, X_test, y_train, y_test, scaler = load_and_split("full")
    model = joblib.load(MODEL_FILE)
    result = permutation_importance(
        model, X_test, y_test, scoring="roc_auc", n_repeats=20, random_state=SEED
    )
    importances = pd.Series(result.importances_mean, index=X_test.columns).sort_values(
        ascending=False
    )
    top = importances.head(10)
    lines = [f"[{SETUP}] top 10 permutation importances ({MODEL_NAME}):"]
    for feature in top.index:
        pos = list(X_test.columns).index(feature)
        lines.append(
            f"  {feature}: {result.importances_mean[pos]:.4f} "
            f"+/- {result.importances_std[pos]:.4f}"
        )
    text = "\n".join(lines) + "\n"
    print(text, end="")
    OUTPUT_FILE.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
