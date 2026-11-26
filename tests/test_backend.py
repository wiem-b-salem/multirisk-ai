import os
import tempfile
from pathlib import Path

TMP_DIR = tempfile.mkdtemp()
os.environ["DATABASE_URL"] = f"sqlite:///{(Path(TMP_DIR) / 'test_backend.db').as_posix()}"
os.environ["ML_API_URL"] = "http://ml-api.test"
os.environ["JWT_SECRET"] = "test-secret-key-with-enough-length-1234567890"

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from src.backend import main as backend
from src.backend.main import (
    Base,
    ComparisonRecord,
    PredictionRecord,
    SessionLocal,
    app,
    engine,
    validate_settings,
)

client = TestClient(app)


@pytest.fixture(autouse=True)
def clean_db():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    yield


async def fake_call_ml_api(method, path, json_body=None):
    if path.startswith("/schema/"):
        return httpx.Response(200, json=[{"name": "age"}])
    disease = path.rsplit("/", 1)[-1]
    if path.startswith("/compare/"):
        return httpx.Response(
            200,
            json={
                "disease": disease,
                "ml": {
                    "model_name": "rf",
                    "risk_score": 0.8,
                    "risk_level": "high",
                },
                "dl": {
                    "model_name": "mlp",
                    "risk_score": 0.4,
                    "risk_level": "low",
                },
                "comparison": {
                    "score_difference": -0.4,
                    "risk_levels_match": False,
                    "interpretation": "Models disagree.",
                },
                "disclaimer": "disclaimer",
            },
        )
    return httpx.Response(
        200,
        json={
            "disease": disease,
            "model_name": "rf",
            "risk_score": 0.8,
            "risk_level": "high",
            "score_note": "note",
            "top_factors": None,
            "disclaimer": "disclaimer",
        },
    )


def register(email="a@example.com", password="password123"):
    return client.post("/auth/register", json={"email": email, "password": password})


def login(email="a@example.com", password="password123"):
    return client.post("/auth/login", json={"email": email, "password": password})


def auth_headers(token):
    return {"Authorization": f"Bearer {token}"}


def test_register_login_predict_history(monkeypatch):
    monkeypatch.setattr(backend, "call_ml_api", fake_call_ml_api)
    assert register().status_code == 201
    login_response = login()
    assert login_response.status_code == 200
    token = login_response.json()["access_token"]

    response = client.post(
        "/predict/cvd", json={"features": {"age": 50}}, headers=auth_headers(token)
    )
    assert response.status_code == 200
    assert response.json()["record_id"] is not None

    history = client.get("/history", headers=auth_headers(token))
    assert history.status_code == 200
    assert len(history.json()) == 1
    assert history.json()[0]["disease"] == "cvd"


def test_predict_requires_auth(monkeypatch):
    monkeypatch.setattr(backend, "call_ml_api", fake_call_ml_api)
    response = client.post("/predict/cvd", json={"features": {"age": 50}})
    assert response.status_code == 401


def test_duplicate_register_returns_409():
    assert register().status_code == 201
    assert register().status_code == 409


def test_ml_api_422_passthrough(monkeypatch):
    async def fake_422(method, path, json_body=None):
        return httpx.Response(422, json={"detail": "bad features"})

    monkeypatch.setattr(backend, "call_ml_api", fake_422)
    register()
    token = login().json()["access_token"]
    response = client.post(
        "/predict/cvd", json={"features": {"age": 50}}, headers=auth_headers(token)
    )
    assert response.status_code == 422
    assert response.json()["detail"] == "bad features"


def test_history_is_per_user(monkeypatch):
    monkeypatch.setattr(backend, "call_ml_api", fake_call_ml_api)
    register("one@example.com")
    register("two@example.com")
    token_one = login("one@example.com").json()["access_token"]
    token_two = login("two@example.com").json()["access_token"]

    client.post(
        "/predict/cvd", json={"features": {"age": 50}}, headers=auth_headers(token_one)
    )
    client.post(
        "/predict/diabetes",
        json={"features": {"age": 50}},
        headers=auth_headers(token_two),
    )

    one_history = client.get("/history", headers=auth_headers(token_one)).json()
    two_history = client.get("/history", headers=auth_headers(token_two)).json()
    assert [record["disease"] for record in one_history] == ["cvd"]
    assert [record["disease"] for record in two_history] == ["diabetes"]


def test_record_stores_no_features(monkeypatch):
    monkeypatch.setattr(backend, "call_ml_api", fake_call_ml_api)
    register()
    token = login().json()["access_token"]
    client.post(
        "/predict/cvd",
        json={"features": {"age": 123456}},
        headers=auth_headers(token),
    )

    assert "features" not in PredictionRecord.__table__.columns.keys()
    with SessionLocal() as db:
        record = db.scalar(select(PredictionRecord))
        assert record is not None
        assert "features" not in record.__dict__
        assert all("123456" not in str(value) for value in record.__dict__.values())


def test_compare_history_stores_both_results_without_features(monkeypatch):
    monkeypatch.setattr(backend, "call_ml_api", fake_call_ml_api)
    register()
    token = login().json()["access_token"]
    response = client.post(
        "/compare/cvd", json={"features": {"age": 123456}}, headers=auth_headers(token)
    )
    assert response.status_code == 200
    assert response.json()["record_id"] is not None

    history = client.get("/history", headers=auth_headers(token)).json()
    assert len(history) == 1
    assert history[0]["record_type"] == "comparison"
    assert history[0]["ml_risk_level"] == "high"
    assert history[0]["dl_risk_level"] == "low"
    assert "features" not in ComparisonRecord.__table__.columns.keys()
    with SessionLocal() as db:
        record = db.scalar(select(ComparisonRecord))
        assert record is not None
        assert "123456" not in str(record.__dict__)


def test_compare_requires_auth(monkeypatch):
    monkeypatch.setattr(backend, "call_ml_api", fake_call_ml_api)
    response = client.post("/compare/cvd", json={"features": {"age": 50}})
    assert response.status_code == 401


def test_validate_settings_production_default_secret_raises():
    with pytest.raises(RuntimeError):
        validate_settings("production", "dev-secret-change-me")


def test_validate_settings_production_strong_secret_passes():
    validate_settings("production", "a" * 32)


def test_validate_settings_development_default_secret_passes():
    validate_settings("development", "dev-secret-change-me")


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
