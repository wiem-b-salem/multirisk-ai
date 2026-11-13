import json
import math
from pathlib import Path

import pandas as pd

from src.cardiovascular.preprocess import load_and_split as load_cvd
from src.diabetes.preprocess import COMPACT_COLUMNS
from src.diabetes.preprocess import load_and_split as load_diabetes

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
HELP = json.loads((ROOT / "config" / "field_help.json").read_text())

CVD_VALID = {
    "age": (15, 100),
    "sbp": (60, 280),
    "tobacco": (0, 300),
    "ldl": (0.5, 20),
    "adiposity": (5, 60),
    "typea": (12, 84),
    "obesity": (10, 70),
    "alcohol": (0, 500),
}

DIABETES_VALID = {
    "Age": (18, 100),
    "BMI": (10, 70),
    "FastingBloodSugar": (40, 600),
    "HbA1c": (3, 18),
    "SystolicBP": (60, 260),
    "DiastolicBP": (30, 160),
}

POPULATION_NOTES = {
    "cvd": (
        "Trained on 462 men aged 15-64 from a high-risk region of South Africa "
        "(case-control sample). Not validated for women or for people outside this "
        "age range."
    ),
    "diabetes_compact": "Trained on a synthetic dataset; not validated on real patients.",
}

SETUPS = {
    "cvd": {
        "data": ROOT / "data" / "cardiovascular" / "CHDdata.csv",
        "target": "chd",
        "drop": [],
        "family_history": "famhist",
        "dir": "cardiovascular",
        "help": "cvd",
        "feature_columns": lambda: list(load_cvd()[0].columns),
        "model_name": "logreg",
    },
    "diabetes_compact": {
        "data": ROOT / "data" / "diabetes" / "diabetes_data.csv",
        "target": "Diagnosis",
        "drop": ["PatientID", "DoctorInCharge"],
        "family_history": "FamilyHistoryDiabetes",
        "dir": "diabetes",
        "fields": COMPACT_COLUMNS,
        "help": "diabetes",
        "feature_columns": lambda: list(load_diabetes("compact")[0].columns),
        "model_name": "rf",
    },
}


def valid_range(disease, name, train_min, train_max, is_int):
    if disease == "cvd":
        return CVD_VALID[name]
    if name in DIABETES_VALID:
        return DIABETES_VALID[name]
    width = train_max - train_min
    low = train_min - 0.25 * width
    high = train_max + 0.25 * width
    if is_int:
        return (math.floor(low), math.ceil(high))
    return (low, high)


def field_schema(name, series, disease, help_entry):
    entry = {
        "name": name,
        "label": help_entry.get("label", name),
        "unit": help_entry.get("unit", ""),
        "help": help_entry.get("help", ""),
    }
    if not pd.api.types.is_numeric_dtype(series):
        allowed = ["Absent", "Present"] if name == "famhist" else sorted(series.dropna().unique())
        entry.update(
            {
                "type": "category",
                "train_min": None,
                "train_max": None,
                "valid_min": None,
                "valid_max": None,
                "allowed_values": allowed,
            }
        )
        return entry
    is_int = pd.api.types.is_integer_dtype(series)
    kind = "int" if is_int else "float"
    train_min = int(series.min()) if is_int else float(series.min())
    train_max = int(series.max()) if is_int else float(series.max())
    entry.update({"type": kind, "train_min": train_min, "train_max": train_max})
    if series.nunique() <= 2:
        entry["allowed_values"] = [
            int(v) if is_int else float(v) for v in sorted(series.dropna().unique())
        ]
        entry["valid_min"] = None
        entry["valid_max"] = None
        return entry
    valid_min, valid_max = valid_range(disease, name, train_min, train_max, is_int)
    entry["valid_min"] = valid_min
    entry["valid_max"] = valid_max
    return entry


def build_schema(name, setup):
    df = pd.read_csv(setup["data"])
    if "fields" in setup:
        fields = setup["fields"]
    else:
        fields = [c for c in df.columns if c != setup["target"] and c not in setup["drop"]]
    help_map = HELP[setup["help"]]
    return {
        "raw_fields": [
            field_schema(c, df[c], name, help_map.get(c, {})) for c in fields
        ],
        "family_history_field": setup["family_history"],
        "population_note": POPULATION_NOTES[name],
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


if __name__ == "__main__":
    main()
