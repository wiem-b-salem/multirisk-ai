import os
import tempfile
from pathlib import Path

TMP_DIR = tempfile.mkdtemp()
os.environ["DATABASE_URL"] = f"sqlite:///{(Path(TMP_DIR) / 'test_liver_backend.db').as_posix()}"
os.environ["ML_API_URL"] = "http://ml-api.test"
os.environ["JWT_SECRET"] = "test-secret-key-with-enough-length-1234567890"

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from src.backend import main as backend
from src.backend.main import Base, PredictionRecord, SessionLocal, app, engine

client = TestClient(app)


@pytest.fixture(autouse=True)
def clean_db():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    yield


async def fake_call_ml_api(method, path, json_body=None):
    if path.startswith("/schema/"):
        return httpx.Response(200, json=[{"name": "Age"}])
    disease = path.rsplit("/", 1)[-1]
    return httpx.Response(
        200,
        json={
            "disease": disease,
            "model_name": "rf",
            "risk_score": 0.42,
            "risk_level": "medium",
            "score_note": "note",
            "top_factors": None,
            "disclaimer": "disclaimer",
            "model_note": "model note",
            "population_note": "population note",
        },
    )


def register():
    return client.post(
        "/auth/register", json={"email": "liver@example.com", "password": "password123"}
    )


def login():
    return client.post(
        "/auth/login", json={"email": "liver@example.com", "password": "password123"}
    )


def test_schema_liver(monkeypatch):
    calls = []

    async def fake(method, path, json_body=None):
        calls.append((method, path))
        return httpx.Response(200, json=[{"name": "GeneticRisk"}])

    monkeypatch.setattr(backend, "call_ml_api", fake)
    response = client.get("/schema/liver")
    assert response.status_code == 200
    assert calls == [("GET", "/schema/liver")]


def test_predict_liver(monkeypatch):
    monkeypatch.setattr(backend, "call_ml_api", fake_call_ml_api)
    assert register().status_code == 201
    token = login().json()["access_token"]
    response = client.post(
        "/predict/liver",
        json={"features": {"GeneticRisk": 2}},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["disease"] == "liver"
    assert data["risk_level"] == "medium"
    assert data["model_name"] == "rf"
    assert data["record_id"] is not None

    history = client.get("/history", headers={"Authorization": f"Bearer {token}"})
    assert history.status_code == 200
    assert [record["disease"] for record in history.json()] == ["liver"]
    with SessionLocal() as db:
        stored = db.scalar(select(PredictionRecord))
        assert stored is not None
        assert stored.disease == "liver"
        assert "features" not in stored.__dict__
