import os
import tempfile
from pathlib import Path

TMP_DIR = tempfile.mkdtemp()
os.environ.setdefault(
    "DATABASE_URL", f"sqlite:///{(Path(TMP_DIR) / 'test_backend_profile.db').as_posix()}"
)
os.environ.setdefault("ML_API_URL", "http://ml-api.test")
os.environ.setdefault("JWT_SECRET", "test-secret-key-with-enough-length-1234567890")

import httpx
import pytest
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient
from sqlalchemy import select

from src.backend import main as backend
from src.backend.main import (
    Base,
    PredictionRecord,
    ProfileData,
    SessionLocal,
    app,
    engine,
    validate_profile_key,
)

client = TestClient(app)


@pytest.fixture(autouse=True)
def clean_db():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    yield


def register(email="profile@example.com", password="password123"):
    return client.post("/auth/register", json={"email": email, "password": password})


def login(email="profile@example.com", password="password123"):
    return client.post("/auth/login", json={"email": email, "password": password})


def auth_headers(token):
    return {"Authorization": f"Bearer {token}"}


def token_for(email="profile@example.com"):
    return login(email).json()["access_token"]


def test_profile_requires_authentication():
    assert client.get("/profile").status_code == 401
    assert client.put("/profile", json={"data": {"shared.age": 40}}).status_code == 401
    assert client.delete("/profile").status_code == 401


def test_new_profile_is_empty():
    register()
    response = client.get("/profile", headers=auth_headers(token_for()))
    assert response.status_code == 200
    assert response.json() == {"data": {}, "updated_at": None}


def test_put_then_get_returns_exact_data():
    register()
    headers = auth_headers(token_for())
    data = {"shared.age": 40, "cvd.famhist": "Present", "smoking.status": None}
    response = client.put("/profile", json={"data": data}, headers=headers)
    assert response.status_code == 200
    assert response.json()["updated_at"] is not None
    fetched = client.get("/profile", headers=headers)
    assert fetched.status_code == 200
    assert fetched.json()["data"] == data


def test_second_put_replaces_first():
    register()
    headers = auth_headers(token_for())
    first = {"shared.age": 40}
    second = {"diabetes.HbA1c": 5.5, "liver.Smoking": False}
    assert client.put("/profile", json={"data": first}, headers=headers).status_code == 200
    assert client.put("/profile", json={"data": second}, headers=headers).status_code == 200
    assert client.get("/profile", headers=headers).json()["data"] == second


def test_profiles_are_per_user():
    register("one@example.com")
    register("two@example.com")
    one = auth_headers(token_for("one@example.com"))
    two = auth_headers(token_for("two@example.com"))
    client.put("/profile", json={"data": {"shared.age": 40}}, headers=one)
    client.put("/profile", json={"data": {"shared.age": 70}}, headers=two)
    assert client.get("/profile", headers=one).json()["data"] == {"shared.age": 40}
    assert client.get("/profile", headers=two).json()["data"] == {"shared.age": 70}


def test_delete_profile_is_idempotent_and_clears_data():
    register()
    headers = auth_headers(token_for())
    client.put("/profile", json={"data": {"shared.age": 40}}, headers=headers)
    assert client.delete("/profile", headers=headers).status_code == 204
    assert client.delete("/profile", headers=headers).status_code == 204
    assert client.get("/profile", headers=headers).json() == {"data": {}, "updated_at": None}


def test_profile_key_outside_whitelist_returns_422():
    register()
    headers = auth_headers(token_for())
    response = client.put(
        "/profile", json={"data": {"other.age": 40}}, headers=headers
    )
    assert response.status_code == 422


def test_profile_list_value_returns_422():
    register()
    headers = auth_headers(token_for())
    response = client.put(
        "/profile", json={"data": {"shared.age": [40]}}, headers=headers
    )
    assert response.status_code == 422


def test_101_keys_return_422():
    register()
    headers = auth_headers(token_for())
    data = {f"shared.k{index}": 1 for index in range(101)}
    response = client.put("/profile", json={"data": data}, headers=headers)
    assert response.status_code == 422


def test_payload_over_limit_returns_413():
    register()
    headers = auth_headers(token_for())
    response = client.put(
        "/profile", json={"data": {"shared.note": "x" * 11000}}, headers=headers
    )
    assert response.status_code == 413


def test_encrypted_column_contains_no_plaintext():
    user_id = register().json()["id"]
    headers = auth_headers(token_for())
    data = {"shared.note": "plaintext-marker-12345", "shared.count": 987654}
    assert client.put("/profile", json={"data": data}, headers=headers).status_code == 200
    with SessionLocal() as db:
        row = db.scalar(select(ProfileData).where(ProfileData.user_id == user_id))
        assert row is not None
        assert "plaintext-marker-12345" not in row.data_encrypted
        assert "987654" not in row.data_encrypted


def test_prediction_history_still_has_no_inputs(monkeypatch):
    async def fake_call_ml_api(method, path, json_body=None):
        return httpx.Response(
            200,
            json={
                "disease": path.rsplit("/", 1)[-1],
                "risk_score": 0.8,
                "risk_level": "high",
            },
        )

    monkeypatch.setattr(backend, "call_ml_api", fake_call_ml_api)
    register()
    headers = auth_headers(token_for())
    marker = "prediction-marker-98765"
    response = client.post(
        "/predict/cvd", json={"features": {"age": marker}}, headers=headers
    )
    assert response.status_code == 200
    history = client.get("/history", headers=headers)
    assert history.status_code == 200
    assert all(
        "feature" not in key.lower()
        for record in history.json()
        for key in record
    )
    with SessionLocal() as db:
        record = db.scalar(select(PredictionRecord))
        assert record is not None
        assert all(marker not in str(value) for value in record.__dict__.values())


def test_validate_profile_key_production_missing_raises():
    with pytest.raises(RuntimeError):
        validate_profile_key("production", None)


def test_validate_profile_key_production_invalid_raises():
    with pytest.raises(RuntimeError):
        validate_profile_key("production", "not-a-fernet-key")


def test_validate_profile_key_generated_key_passes():
    validate_profile_key("production", Fernet.generate_key().decode())


def test_validate_profile_key_development_missing_passes():
    validate_profile_key("development", None)
