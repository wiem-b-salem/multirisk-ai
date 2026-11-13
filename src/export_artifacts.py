import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.cardiovascular.preprocess import load_and_split as load_cvd
from src.diabetes.preprocess import load_and_split as load_diabetes

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
CONFIG_DIR = ROOT / "config"

SETUPS = {
    "cvd": {
        "data": ROOT / "data" / "cardiovascular" / "CHDdata.csv",
        "target": "chd",
        "drop": [],
        "family_history": "famhist",
        "dir": "cardiovascular",
        "feature_columns": lambda: list(load_cvd()[0].columns),
        "model_name": "logreg",
    },
    "diabetes_full": {
        "data": ROOT / "data" / "diabetes" / "diabetes_data.csv",
        "target": "Diagnosis",
        "drop": ["PatientID", "DoctorInCharge"],
        "family_history": "FamilyHistoryDiabetes",
        "dir": "diabetes",
        "feature_columns": lambda: list(load_diabetes("full")[0].columns),
        "model_name": "rf",
    },
}


def training_rows(df, target):
    train_idx, _ = train_test_split(
        df.index, test_size=0.2, stratify=df[target], random_state=SEED
    )
    return df.loc[train_idx]


def field_schema(name, series):
    if not pd.api.types.is_numeric_dtype(series):
        allowed = ["Absent", "Present"] if name == "famhist" else sorted(series.dropna().unique())
        return {
            "name": name,
            "type": "category",
            "min": None,
            "max": None,
            "allowed_values": allowed,
        }
    if name == "Ethnicity":
        allowed = sorted(int(v) for v in series.dropna().unique())
        return {
            "name": name,
            "type": "category",
            "min": int(series.min()),
            "max": int(series.max()),
            "allowed_values": allowed,
        }
    is_int = pd.api.types.is_integer_dtype(series)
    kind = "int" if is_int else "float"
    minimum = int(series.min()) if is_int else float(series.min())
    maximum = int(series.max()) if is_int else float(series.max())
    entry = {"name": name, "type": kind, "min": minimum, "max": maximum}
    if series.nunique() <= 2:
        entry["allowed_values"] = [
            int(v) if is_int else float(v) for v in sorted(series.dropna().unique())
        ]
    return entry


def build_schema(name, setup):
    df = pd.read_csv(setup["data"])
    raw = training_rows(df, setup["target"])
    fields = [c for c in df.columns if c != setup["target"] and c not in setup["drop"]]
    return {
        "raw_fields": [field_schema(c, raw[c]) for c in fields],
        "family_history_field": setup["family_history"],
        "feature_columns": setup["feature_columns"](),
        "model_file": f"final_{name}.joblib",
        "scaler_file": f"scaler_{name}.joblib",
        "model_name": setup["model_name"],
    }


def main():
    for name, setup in SETUPS.items():
        schema = build_schema(name, setup)
        path = MODELS_DIR / setup["dir"] / f"schema_{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(schema, indent=2))
        print(f"Wrote {path}")
        print(json.dumps(schema, indent=2))

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    thresholds_path = CONFIG_DIR / "risk_thresholds.json"
    thresholds = {
        "low_max": 0.30,
        "high_min": 0.60,
        "note": "Provisional thresholds; must be chosen from calibration/ROC analysis.",
    }
    thresholds_path.write_text(json.dumps(thresholds, indent=2))
    print(f"Wrote {thresholds_path}")
    print(json.dumps(thresholds, indent=2))


if __name__ == "__main__":
    main()
