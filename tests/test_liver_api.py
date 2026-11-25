import json
from pathlib import Path

from fastapi.testclient import TestClient

from src.ml_api.main import app

ROOT = Path(__file__).resolve().parents[1]
FOLDERS = {"liver": ("liver_full", "liver")}
client = TestClient(app)


def load_schema(disease):
    setup, folder = FOLDERS[disease]
    return json.loads((ROOT / "models" / folder / f"schema_{setup}.json").read_text())


def make_payload(disease):
    features = {}
    for field in load_schema(disease)["raw_fields"]:
        if field.get("allowed_values"):
            features[field["name"]] = field["allowed_values"][0]
        elif field["type"] == "int":
            features[field["name"]] = (field["train_min"] + field["train_max"]) // 2
        else:
            features[field["name"]] = (field["train_min"] + field["train_max"]) / 2
    return {"features": features}


def test_valid_liver_prediction():
    response = client.post("/predict/liver", json=make_payload("liver"))
    assert response.status_code == 200
    data = response.json()
    assert 0 <= data["risk_score"] <= 1
    assert data["risk_level"] in {"low", "medium", "high"}
    assert data["population_note"]
    assert "LiverFunctionTest" in data["model_note"]
    assert "AlcoholConsumption" in data["model_note"]


def test_missing_genetic_risk():
    payload = make_payload("liver")
    del payload["features"]["GeneticRisk"]
    response = client.post("/predict/liver", json=payload)
    assert response.status_code == 422


def test_invalid_genetic_risk():
    payload = make_payload("liver")
    payload["features"]["GeneticRisk"] = 3
    response = client.post("/predict/liver", json=payload)
    assert response.status_code == 422


def test_value_out_of_valid_range():
    payload = make_payload("liver")
    payload["features"]["Age"] = 150
    response = client.post("/predict/liver", json=payload)
    assert response.status_code == 422

    payload = make_payload("liver")
    payload["features"]["LiverFunctionTest"] = 400
    response = client.post("/predict/liver", json=payload)
    assert response.status_code == 422


def test_age_out_of_training_range():
    payload = make_payload("liver")
    payload["features"]["Age"] = 85
    response = client.post("/predict/liver", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert any(item["field"] == "Age" for item in data["out_of_training_range"])
    assert "range_warning" in data


def test_schema():
    response = client.get("/schema/liver")
    assert response.status_code == 200
    schema = response.json()
    assert schema["family_history_field"] == "GeneticRisk"
    assert schema["family_history_box_title"] == "Genetic risk (required)"
    assert schema["model_file"] == "final_liver_full.joblib"
    assert schema["model_name"] in {"logreg", "rf"}
