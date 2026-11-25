import json
from pathlib import Path

import pandas as pd

from src.liver.preprocess import DATA_PATH, FULL_COLUMNS

ROOT = Path(__file__).resolve().parents[2]
HELP = json.loads((ROOT / "config" / "field_help.json").read_text())
SCHEMA_PATH = ROOT / "models" / "liver" / "schema_liver_full.json"
VALID = {
    "Age": (18, 110),
    "BMI": (10, 70),
    "AlcoholConsumption": (0, 100),
    "PhysicalActivity": (0, 10),
    "LiverFunctionTest": (0, 300),
}
BINARY_FIELDS = ["Gender", "Smoking", "Diabetes", "Hypertension"]
INT_ALLOWED = {"GeneticRisk": [0, 1, 2]}
POPULATION_NOTE = (
    "Trained on a synthetic dataset of 1,700 people aged 20-80; "
    "not validated on real patients. The genetic risk level is not a family-history question."
)


def field_schema(name, series, help_entry):
    is_int = pd.api.types.is_integer_dtype(series)
    entry = {
        "name": name,
        "label": help_entry.get("label", name),
        "unit": help_entry.get("unit", ""),
        "help": help_entry.get("help", ""),
        "type": "int" if is_int else "float",
        "train_min": int(series.min()) if is_int else float(series.min()),
        "train_max": int(series.max()) if is_int else float(series.max()),
    }
    if name in BINARY_FIELDS:
        entry["allowed_values"] = [0, 1]
    elif name in INT_ALLOWED:
        entry["allowed_values"] = INT_ALLOWED[name]
    if "allowed_values" in entry:
        entry["valid_min"] = None
        entry["valid_max"] = None
    else:
        entry["valid_min"], entry["valid_max"] = VALID[name]
    return entry


def main():
    df = pd.read_csv(DATA_PATH)
    summary = pd.read_csv(ROOT / "results" / "liver_summary.csv")
    subset = summary[summary["setup"] == "liver_full"]
    model_name = subset.loc[subset["cv_roc_auc_mean"].idxmax(), "model"]
    help_map = HELP["liver"]
    schema = {
        "raw_fields": [
            field_schema(name, df[name], help_map.get(name, {})) for name in FULL_COLUMNS
        ],
        "family_history_field": "GeneticRisk",
        "family_history_box_title": "Genetic risk (required)",
        "population_note": POPULATION_NOTE,
        "feature_columns": FULL_COLUMNS,
        "model_file": "final_liver_full.joblib",
        "scaler_file": "scaler_liver_full.joblib",
        "model_name": model_name,
    }
    SCHEMA_PATH.parent.mkdir(parents=True, exist_ok=True)
    SCHEMA_PATH.write_text(json.dumps(schema, indent=2))
    print(f"Wrote {SCHEMA_PATH}")
    print(json.dumps(schema, indent=2))


if __name__ == "__main__":
    main()
