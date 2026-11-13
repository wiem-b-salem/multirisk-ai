import json
from pathlib import Path
from typing import Annotated, Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import Body, FastAPI, HTTPException
from pydantic import Field, ValidationError, create_model

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
CONFIG_DIR = ROOT / "config"

DISEASES = {
    "cvd": ("cvd", "cardiovascular"),
    "diabetes": ("diabetes_full", "diabetes"),
}
DISCLAIMER = (
    "This is an estimate for academic demonstration, not a medical diagnosis. "
    "Consult a healthcare professional."
)
MODEL_NOTE = (
    "This model was trained on a synthetic dataset and depends heavily on "
    "HbA1c and fasting blood sugar."
)

app = FastAPI()


def build_features_model(schema):
    fields = {}
    for field in schema["raw_fields"]:
        allowed = field.get("allowed_values")
        if allowed:
            field_type = Literal.__getitem__(tuple(allowed))
        elif field["type"] == "int":
            field_type = Annotated[int, Field(ge=field["min"], le=field["max"])]
        else:
            field_type = Annotated[float, Field(ge=field["min"], le=field["max"])]
        fields[field["name"]] = (field_type, ...)
    return create_model(f"{schema['model_name']}_features", **fields)


def load_artifacts():
    thresholds = json.loads((CONFIG_DIR / "risk_thresholds.json").read_text())
    artifacts = {}
    for disease, (setup, folder) in DISEASES.items():
        schema = json.loads((MODELS_DIR / folder / f"schema_{setup}.json").read_text())
        features_model = build_features_model(schema)
        request_model = create_model(f"{setup}_request", features=(features_model, ...))
        artifacts[disease] = {
            "disease": disease,
            "schema": schema,
            "model": joblib.load(MODELS_DIR / folder / schema["model_file"]),
            "scaler": joblib.load(MODELS_DIR / folder / schema["scaler_file"]),
            "request_model": request_model,
        }
    return artifacts, thresholds


ARTIFACTS, THRESHOLDS = load_artifacts()


def preprocess(state, features):
    schema = state["schema"]
    columns = [field["name"] for field in schema["raw_fields"]]
    frame = pd.DataFrame([features], columns=columns)

    if state["disease"] == "cvd":
        frame["famhist"] = frame["famhist"].map({"Absent": 0, "Present": 1})
    else:
        ethnicity = next(
            field for field in schema["raw_fields"] if field["name"] == "Ethnicity"
        )
        frame["Ethnicity"] = pd.Categorical(
            frame["Ethnicity"], categories=ethnicity["allowed_values"]
        )
        frame = pd.get_dummies(frame, columns=["Ethnicity"], drop_first=True, dtype=int)

    frame = frame.reindex(columns=schema["feature_columns"], fill_value=0)
    scaled = state["scaler"].transform(frame)
    return pd.DataFrame(scaled, columns=schema["feature_columns"])


def risk_level(probability):
    if probability < THRESHOLDS["low_max"]:
        return "low"
    if probability >= THRESHOLDS["high_min"]:
        return "high"
    return "medium"


def factor_list(model, scaled):
    coefficients = model.coef_[0]
    values = scaled.iloc[0].to_numpy()
    contributions = coefficients * values
    order = np.argsort(np.abs(contributions))[::-1][:5]
    return [
        {
            "feature": scaled.columns[index],
            "contribution": float(contributions[index]),
            "direction": "increases" if contributions[index] >= 0 else "decreases",
        }
        for index in order
    ]


@app.get("/health")
def health():
    return {"status": "ok", "diseases": list(ARTIFACTS.keys())}


@app.get("/schema/{disease}")
def schema(disease: str):
    if disease not in ARTIFACTS:
        raise HTTPException(status_code=404, detail=f"Unknown disease: {disease}")
    return ARTIFACTS[disease]["schema"]["raw_fields"]


@app.post("/predict/{disease}")
def predict(disease: str, body: dict = Body(...)):
    if disease not in ARTIFACTS:
        raise HTTPException(status_code=404, detail=f"Unknown disease: {disease}")

    state = ARTIFACTS[disease]
    try:
        payload = state["request_model"].model_validate(body)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors())

    features = payload.features.model_dump()
    scaled = preprocess(state, features)
    model = state["model"]
    probability = float(model.predict_proba(scaled)[0, 1])

    result = {
        "disease": disease,
        "model_name": state["schema"]["model_name"],
        "risk_probability": probability,
        "risk_level": risk_level(probability),
        "top_factors": factor_list(model, scaled)
        if state["schema"]["model_name"] == "logreg"
        else None,
        "disclaimer": DISCLAIMER,
    }
    if disease == "diabetes":
        result["model_note"] = MODEL_NOTE
    return result
