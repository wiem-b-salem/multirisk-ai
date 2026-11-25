import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.alzheimers.preprocess import load_and_split as load_alzheimers
from src.baselines import MODELS
from src.calibrate_thresholds import band_table

SEED = 42
ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "risk_thresholds.json"


def main():
    summary = pd.read_csv(ROOT / "results" / "alzheimers_summary.csv")
    subset = summary[summary["setup"] == "alzheimers_compact"]
    model_name = subset.loc[subset["cv_roc_auc_mean"].idxmax(), "model"]
    X_train, X_test, y_train, y_test, _ = load_alzheimers("compact")
    y = y_train.to_numpy()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = cross_val_predict(
        MODELS[model_name](), X_train, y_train, cv=cv, method="predict_proba"
    )[:, 1]

    candidates = np.unique(scores)
    recalls, specificities = [], []
    for threshold in candidates:
        predicted = scores >= threshold
        tp = np.sum(predicted & (y == 1))
        fn = np.sum(~predicted & (y == 1))
        tn = np.sum(~predicted & (y == 0))
        fp = np.sum(predicted & (y == 0))
        recalls.append(tp / (tp + fn))
        specificities.append(tn / (tn + fp))

    recalls = np.array(recalls)
    specificities = np.array(specificities)
    low_max = float(candidates[recalls >= 0.90].max())
    high_min = float(candidates[specificities >= 0.90].min())
    if low_max >= high_min:
        j = recalls + specificities - 1
        threshold = float(candidates[np.argmax(j)])
        low_max = high_min = threshold
        print(
            f"NOTE: alzheimers low_max >= high_min; using Youden's J threshold "
            f"{threshold:.4f} (two levels: low/high)."
        )

    thresholds = {"low_max": low_max, "high_min": high_min}
    print(f"\nalzheimers out-of-fold (low_max={low_max:.4f}, high_min={high_min:.4f})")
    print(band_table(scores, y, low_max, high_min).to_string(index=False))

    original = json.loads(CONFIG_PATH.read_text())
    config = dict(original)
    config["alzheimers"] = thresholds
    CONFIG_PATH.write_text(json.dumps(config, indent=2))
    updated = json.loads(CONFIG_PATH.read_text())
    print("\nConfig check:")
    for disease in ("cvd", "diabetes"):
        unchanged = updated[disease] == original[disease]
        print(f"{disease} unchanged: {unchanged}")
        assert unchanged
    print(f"Wrote {CONFIG_PATH}")

    model = MODELS[model_name]()
    model.fit(X_train, y_train)
    test_scores = model.predict_proba(X_test)[:, 1]
    print("\nFinal test-set check (frozen thresholds):")
    print(
        band_table(
            test_scores, y_test.to_numpy(), thresholds["low_max"], thresholds["high_min"]
        ).to_string(index=False)
    )


if __name__ == "__main__":
    main()
