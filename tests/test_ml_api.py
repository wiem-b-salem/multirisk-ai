import json
from pathlib import Path

from fastapi.testclient import TestClient

from src.ml_api.main import app

ROOT = Path(__file__).resolve().parents[1]
FOLDERS = {"cvd": ("cvd", "cardiovascular"), "diabetes": ("diabetes_compact", "diabetes")}
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


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_valid_cvd_prediction():
    response = client.post("/predict/cvd", json=make_payload("cvd"))
    assert response.status_code == 200
    score = response.json()["risk_score"]
    assert 0 <= score <= 1
    print(json.dumps(response.json(), indent=2))


def test_response_fields():
    response = client.post("/predict/cvd", json=make_payload("cvd"))
    data = response.json()
    assert "risk_score" in data
    assert "score_note" in data
    assert "factors_note" in data
    for factor in data["top_factors"]:
        assert factor["effect"] in {"above_average", "below_average"}


def test_valid_diabetes_prediction():
    response = client.post("/predict/diabetes", json=make_payload("diabetes"))
    assert response.status_code == 200


def test_cvd_missing_famhist():
    payload = make_payload("cvd")
    del payload["features"]["famhist"]
    response = client.post("/predict/cvd", json=payload)
    assert response.status_code == 422


def test_diabetes_missing_family_history():
    payload = make_payload("diabetes")
    del payload["features"]["FamilyHistoryDiabetes"]
    response = client.post("/predict/diabetes", json=payload)
    assert response.status_code == 422


def test_value_out_of_range():
    payload = make_payload("cvd")
    field = next(
        field
        for field in load_schema("cvd")["raw_fields"]
        if field["type"] in ("int", "float") and not field.get("allowed_values")
    )
    payload["features"][field["name"]] = field["valid_max"] + 1
    response = client.post("/predict/cvd", json=payload)
    assert response.status_code == 422


def test_cvd_age_out_of_training_range():
    payload = make_payload("cvd")
    payload["features"]["age"] = 80
    response = client.post("/predict/cvd", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert any(item["field"] == "age" for item in data["out_of_training_range"])
    assert "range_warning" in data


def test_cvd_age_out_of_valid_range():
    payload = make_payload("cvd")
    payload["features"]["age"] = 150
    response = client.post("/predict/cvd", json=payload)
    assert response.status_code == 422


def test_diabetes_fasting_blood_sugar_out_of_training_range():
    payload = make_payload("diabetes")
    payload["features"]["FastingBloodSugar"] = 300
    response = client.post("/predict/diabetes", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert any(
        item["field"] == "FastingBloodSugar" for item in data["out_of_training_range"]
    )
    assert "range_warning" in data


def test_diabetes_fasting_blood_sugar_out_of_valid_range():
    payload = make_payload("diabetes")
    payload["features"]["FastingBloodSugar"] = 700
    response = client.post("/predict/diabetes", json=payload)
    assert response.status_code == 422


def test_compare_returns_ml_and_dl_results():
    for disease in ("cvd", "diabetes"):
        response = client.post(f"/compare/{disease}", json=make_payload(disease))
        assert response.status_code == 200
        data = response.json()
        assert data["ml"]["model_name"] in {"logreg", "rf"}
        assert data["dl"]["model_name"] == "mlp"
        assert data["dl"]["model_label"] == "Neural Network (MLP)"
        assert 0 <= data["ml"]["risk_score"] <= 1
        assert 0 <= data["dl"]["risk_score"] <= 1
        assert data["ml"]["risk_level"] in {"low", "medium", "high"}
        assert data["dl"]["risk_level"] in {"low", "medium", "high"}
        assert data["comparison"]["risk_levels_match"] in {True, False}
        print(json.dumps(data, indent=2))


def test_compare_unknown_disease():
    response = client.post("/compare/unknown", json=make_payload("cvd"))
    assert response.status_code == 404


def test_unknown_disease():
    response = client.post("/predict/unknown", json=make_payload("cvd"))
    assert response.status_code == 404
