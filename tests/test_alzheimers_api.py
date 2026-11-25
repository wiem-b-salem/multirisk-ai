import json
from pathlib import Path

from fastapi.testclient import TestClient

from src.ml_api.main import app

ROOT = Path(__file__).resolve().parents[1]
FOLDERS = {"alzheimers": ("alzheimers_compact", "alzheimers")}
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


def test_valid_alzheimers_prediction():
    response = client.post("/predict/alzheimers", json=make_payload("alzheimers"))
    assert response.status_code == 200
    data = response.json()
    assert 0 <= data["risk_score"] <= 1
    assert data["risk_level"] in {"low", "medium", "high"}
    assert data["population_note"]


def test_missing_family_history():
    payload = make_payload("alzheimers")
    del payload["features"]["FamilyHistoryAlzheimers"]
    response = client.post("/predict/alzheimers", json=payload)
    assert response.status_code == 422


def test_value_out_of_valid_range():
    payload = make_payload("alzheimers")
    payload["features"]["Age"] = 150
    response = client.post("/predict/alzheimers", json=payload)
    assert response.status_code == 422

    payload = make_payload("alzheimers")
    payload["features"]["MMSE"] = 100
    response = client.post("/predict/alzheimers", json=payload)
    assert response.status_code == 422


def test_age_out_of_training_range():
    payload = make_payload("alzheimers")
    payload["features"]["Age"] = 95
    response = client.post("/predict/alzheimers", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert any(item["field"] == "Age" for item in data["out_of_training_range"])
    assert "range_warning" in data


def test_schema():
    response = client.get("/schema/alzheimers")
    assert response.status_code == 200
    schema = response.json()
    assert schema["family_history_field"] == "FamilyHistoryAlzheimers"
    assert schema["model_file"] == "final_alzheimers_compact.joblib"
    assert schema["model_name"] == "logreg"
