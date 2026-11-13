import json
from pathlib import Path

from fastapi.testclient import TestClient

from src.ml_api.main import app

ROOT = Path(__file__).resolve().parents[1]
FOLDERS = {"cvd": ("cvd", "cardiovascular"), "diabetes": ("diabetes_full", "diabetes")}
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
            features[field["name"]] = (field["min"] + field["max"]) // 2
        else:
            features[field["name"]] = (field["min"] + field["max"]) / 2
    return {"features": features}


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_valid_cvd_prediction():
    response = client.post("/predict/cvd", json=make_payload("cvd"))
    assert response.status_code == 200
    probability = response.json()["risk_probability"]
    assert 0 <= probability <= 1
    print(json.dumps(response.json(), indent=2))


def test_valid_diabetes_prediction():
    response = client.post("/predict/diabetes", json=make_payload("diabetes"))
    assert response.status_code == 200


def test_cvd_missing_famhist():
    payload = make_payload("cvd")
    del payload["features"]["famhist"]
    response = client.post("/predict/cvd", json=payload)
    assert response.status_code == 422


def test_value_out_of_range():
    payload = make_payload("cvd")
    field = next(
        field
        for field in load_schema("cvd")["raw_fields"]
        if field["type"] in ("int", "float") and not field.get("allowed_values")
    )
    payload["features"][field["name"]] = field["max"] + 1
    response = client.post("/predict/cvd", json=payload)
    assert response.status_code == 422


def test_unknown_disease():
    response = client.post("/predict/unknown", json=make_payload("cvd"))
    assert response.status_code == 404
