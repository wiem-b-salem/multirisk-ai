import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.baselines import MODELS
from src.cardiovascular.preprocess import load_and_split as load_cvd
from src.diabetes.preprocess import load_and_split as load_diabetes

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "risk_thresholds.json"
DIABETES_SCHEMA = json.loads(
    (ROOT / "models" / "diabetes" / "schema_diabetes_compact.json").read_text()
)

SETUPS = {
    "cvd": (lambda: load_cvd(), MODELS["logreg"]),
    "diabetes": (lambda: load_diabetes("compact"), MODELS[DIABETES_SCHEMA["model_name"]]),
}


def band_table(scores, y, low_max, high_min):
    if low_max == high_min:
        labels = np.where(scores <= low_max, "low", "high")
        bands = ("low", "high")
    else:
        labels = np.where(
            scores < low_max, "low", np.where(scores < high_min, "medium", "high")
        )
        bands = ("low", "medium", "high")

    rows = []
    for band in bands:
        mask = labels == band
        rows.append(
            {
                "band": band,
                "rows": int(mask.sum()),
                "share": float(mask.mean()),
                "positive_rate": float(y[mask].mean()) if mask.any() else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def main():
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    thresholds = {}
    checks = {}
    for disease, (loader, factory) in SETUPS.items():
        X_train, X_test, y_train, y_test, _ = loader()
        y = y_train.to_numpy()
        scores = cross_val_predict(
            factory(), X_train, y_train, cv=cv, method="predict_proba"
        )
        if scores.ndim == 2:
            scores = scores[:, 1]

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
                f"NOTE: {disease} low_max >= high_min; using Youden's J threshold "
                f"{threshold:.4f} (two levels: low/high)."
            )

        thresholds[disease] = {"low_max": low_max, "high_min": high_min}
        checks[disease] = (X_train, X_test, y_train, y_test, factory)
        print(f"\n{disease} out-of-fold (low_max={low_max:.4f}, high_min={high_min:.4f})")
        print(band_table(scores, y, low_max, high_min).to_string(index=False))

    config = {
        "cvd": thresholds["cvd"],
        "diabetes": thresholds["diabetes"],
        "note": (
            "Chosen from out-of-fold predictions on the training set. "
            "cvd: recall>=0.90 below low_max, specificity>=0.90 above high_min. "
            "diabetes: single threshold maximizing Youden's J (two risk levels)."
        ),
    }
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(config, indent=2))
    print(f"\nWrote {CONFIG_PATH}")
    print(json.dumps(config, indent=2))

    print("\nFinal test-set check (frozen thresholds):")
    for disease, (X_train, X_test, y_train, y_test, factory) in checks.items():
        model = factory()
        model.fit(X_train, y_train)
        scores = model.predict_proba(X_test)[:, 1]
        frozen = thresholds[disease]
        print(
            f"\n{disease} test "
            f"(low_max={frozen['low_max']:.4f}, high_min={frozen['high_min']:.4f})"
        )
        print(
            band_table(
                scores, y_test.to_numpy(), frozen["low_max"], frozen["high_min"]
            ).to_string(index=False)
        )


if __name__ == "__main__":
    main()
