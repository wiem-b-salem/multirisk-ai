import json
import os
import re
from datetime import datetime, timedelta, timezone

import bcrypt
import httpx
import jwt
from cryptography.fernet import Fernet
from fastapi import Body, Depends, FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field
from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./backend.db")
ML_API_URL = os.getenv("ML_API_URL", "http://127.0.0.1:8000")
JWT_SECRET = os.getenv("JWT_SECRET", "dev-secret-change-me")
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS", "http://localhost:5173,http://localhost:3000"
    ).split(",")
    if origin.strip()
]
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_MINUTES = 60
DEV_PROFILE_ENCRYPTION_KEY = "XUckKVMxIZEv_XCr11EzWGF-cSQ2TM8I5ciCirKCCCk="
PROFILE_ENCRYPTION_KEY = os.getenv("PROFILE_ENCRYPTION_KEY")
PROFILE_KEY_PATTERN = re.compile(
    r"^(shared|cvd|diabetes|alzheimers|liver|smoking)\.[A-Za-z0-9_]{1,40}$"
)

if ENVIRONMENT != "production" and not PROFILE_ENCRYPTION_KEY:
    PROFILE_ENCRYPTION_KEY = DEV_PROFILE_ENCRYPTION_KEY


def validate_settings(environment: str, jwt_secret: str) -> None:
    if environment == "production" and (
        jwt_secret == "dev-secret-change-me" or len(jwt_secret) < 32
    ):
        raise RuntimeError(
            "JWT_SECRET must be a non-default value of at least 32 characters "
            "when ENVIRONMENT=production."
        )


def validate_profile_key(environment: str, key: str | None) -> None:
    if environment != "production":
        return
    if not key:
        raise RuntimeError("PROFILE_ENCRYPTION_KEY must be set when ENVIRONMENT=production.")
    try:
        Fernet(key.encode())
    except (TypeError, ValueError):
        raise RuntimeError("PROFILE_ENCRYPTION_KEY must be a valid Fernet key.")


validate_settings(ENVIRONMENT, JWT_SECRET)
validate_profile_key(ENVIRONMENT, PROFILE_ENCRYPTION_KEY)

if JWT_SECRET == "dev-secret-change-me":
    print("WARNING: using default JWT_SECRET; set JWT_SECRET in production.")

if PROFILE_ENCRYPTION_KEY == DEV_PROFILE_ENCRYPTION_KEY:
    print("WARNING: using built-in development PROFILE_ENCRYPTION_KEY.")

fernet = Fernet(PROFILE_ENCRYPTION_KEY.encode())

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String, unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )
    records: Mapped[list["PredictionRecord"]] = relationship(back_populates="user")


class PredictionRecord(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    disease: Mapped[str] = mapped_column(String)
    risk_level: Mapped[str] = mapped_column(String)
    risk_score: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )
    user: Mapped[User] = relationship(back_populates="records")


class ComparisonRecord(Base):
    __tablename__ = "comparison_predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    disease: Mapped[str] = mapped_column(String)
    ml_risk_level: Mapped[str] = mapped_column(String)
    ml_risk_score: Mapped[float] = mapped_column(Float)
    dl_risk_level: Mapped[str] = mapped_column(String)
    dl_risk_score: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )


class ProfileData(Base):
    __tablename__ = "profiles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), unique=True)
    data_encrypted: Mapped[str] = mapped_column(Text)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )


Base.metadata.create_all(bind=engine)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

bearer = HTTPBearer(auto_error=False)


@app.get("/health")
def health():
    return {"status": "ok"}


class RegisterRequest(BaseModel):
    email: str
    password: str = Field(min_length=8)


class LoginRequest(BaseModel):
    email: str
    password: str


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(password.encode(), password_hash.encode())


def create_token(user_id: int) -> str:
    payload = {
        "sub": str(user_id),
        "exp": datetime.now(timezone.utc) + timedelta(minutes=JWT_EXPIRE_MINUTES),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer),
    db=Depends(get_db),
):
    if credentials is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        payload = jwt.decode(
            credentials.credentials, JWT_SECRET, algorithms=[JWT_ALGORITHM]
        )
        user_id = int(payload["sub"])
    except (jwt.PyJWTError, KeyError, ValueError):
        raise HTTPException(status_code=401, detail="Invalid token")
    user = db.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid token")
    return user


async def call_ml_api(method: str, path: str, json_body=None):
    async with httpx.AsyncClient(base_url=ML_API_URL, timeout=10) as client:
        return await client.request(method, path, json=json_body)


def upstream_detail(response):
    try:
        return response.json().get("detail", response.text)
    except ValueError:
        return response.text


@app.post("/auth/register", status_code=201)
def register(body: RegisterRequest, db=Depends(get_db)):
    existing = db.scalar(select(User).where(User.email == body.email))
    if existing is not None:
        raise HTTPException(status_code=409, detail="Email already registered")
    user = User(email=body.email, password_hash=hash_password(body.password))
    db.add(user)
    db.commit()
    db.refresh(user)
    return {"id": user.id, "email": user.email}


@app.post("/auth/login")
def login(body: LoginRequest, db=Depends(get_db)):
    user = db.scalar(select(User).where(User.email == body.email))
    if user is None or not verify_password(body.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {"access_token": create_token(user.id), "token_type": "bearer"}


@app.get("/me")
def me(user: User = Depends(get_current_user)):
    return {"id": user.id, "email": user.email, "created_at": user.created_at}


class ProfileRequest(BaseModel):
    data: dict


def validate_profile_data(data: dict) -> bytes:
    encoded = json.dumps(data, separators=(",", ":")).encode()
    if len(encoded) > 10240:
        raise HTTPException(status_code=413, detail="Profile data too large")
    if len(data) > 100:
        raise HTTPException(status_code=422, detail="Invalid profile data")
    for key, value in data.items():
        if not isinstance(key, str) or not PROFILE_KEY_PATTERN.fullmatch(key):
            raise HTTPException(status_code=422, detail="Invalid profile data")
        if value is None or isinstance(value, (int, float, bool)):
            continue
        if isinstance(value, str) and len(value) <= 40:
            continue
        raise HTTPException(status_code=422, detail="Invalid profile data")
    return encoded


@app.get("/profile")
def get_profile(user: User = Depends(get_current_user), db=Depends(get_db)):
    row = db.scalar(select(ProfileData).where(ProfileData.user_id == user.id))
    if row is None:
        return {"data": {}, "updated_at": None}
    try:
        data = json.loads(fernet.decrypt(row.data_encrypted.encode()).decode())
    except Exception:
        raise HTTPException(status_code=500, detail="Profile could not be read")
    return {
        "data": data,
        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
    }


@app.put("/profile")
def put_profile(
    body: ProfileRequest,
    user: User = Depends(get_current_user),
    db=Depends(get_db),
):
    encrypted = fernet.encrypt(validate_profile_data(body.data)).decode()
    now = datetime.now(timezone.utc)
    row = db.scalar(select(ProfileData).where(ProfileData.user_id == user.id))
    if row is None:
        row = ProfileData(user_id=user.id, data_encrypted=encrypted, updated_at=now)
        db.add(row)
    else:
        row.data_encrypted = encrypted
        row.updated_at = now
    db.commit()
    db.refresh(row)
    return {"updated_at": row.updated_at.isoformat()}


@app.delete("/profile", status_code=204)
def delete_profile(user: User = Depends(get_current_user), db=Depends(get_db)):
    row = db.scalar(select(ProfileData).where(ProfileData.user_id == user.id))
    if row is not None:
        db.delete(row)
        db.commit()
    return Response(status_code=204)


@app.get("/schema/{disease}")
async def schema(disease: str):
    try:
        response = await call_ml_api("GET", f"/schema/{disease}")
    except httpx.RequestError:
        raise HTTPException(status_code=503, detail="ML API unreachable")
    if response.status_code != 200:
        raise HTTPException(
            status_code=response.status_code, detail=upstream_detail(response)
        )
    return response.json()


@app.post("/predict/{disease}")
async def predict(
    disease: str,
    body: dict = Body(...),
    user: User = Depends(get_current_user),
    db=Depends(get_db),
):
    try:
        response = await call_ml_api("POST", f"/predict/{disease}", body)
    except httpx.RequestError:
        raise HTTPException(status_code=503, detail="ML API unreachable")
    if response.status_code in (422, 404):
        raise HTTPException(
            status_code=response.status_code, detail=upstream_detail(response)
        )
    if response.status_code != 200:
        raise HTTPException(status_code=502, detail="ML API error")
    data = response.json()
    record = PredictionRecord(
        user_id=user.id,
        disease=data["disease"],
        risk_level=data["risk_level"],
        risk_score=data["risk_score"],
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return {**data, "record_id": record.id}


@app.post("/compare/{disease}")
async def compare(
    disease: str,
    body: dict = Body(...),
    user: User = Depends(get_current_user),
    db=Depends(get_db),
):
    try:
        response = await call_ml_api("POST", f"/compare/{disease}", body)
    except httpx.RequestError:
        raise HTTPException(status_code=503, detail="ML API unreachable")
    if response.status_code in (422, 404):
        raise HTTPException(
            status_code=response.status_code, detail=upstream_detail(response)
        )
    if response.status_code != 200:
        raise HTTPException(status_code=502, detail="ML API error")
    data = response.json()
    ml = data["ml"]
    dl = data["dl"]
    record = ComparisonRecord(
        user_id=user.id,
        disease=data["disease"],
        ml_risk_level=ml["risk_level"],
        ml_risk_score=ml["risk_score"],
        dl_risk_level=dl["risk_level"],
        dl_risk_score=dl["risk_score"],
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return {**data, "record_id": record.id}


@app.get("/history")
def history(user: User = Depends(get_current_user), db=Depends(get_db)):
    predictions = db.scalars(
        select(PredictionRecord)
        .where(PredictionRecord.user_id == user.id)
        .order_by(PredictionRecord.created_at.desc(), PredictionRecord.id.desc())
    ).all()
    comparisons = db.scalars(
        select(ComparisonRecord)
        .where(ComparisonRecord.user_id == user.id)
        .order_by(ComparisonRecord.created_at.desc(), ComparisonRecord.id.desc())
    ).all()
    rows = [
        {
            "id": record.id,
            "record_type": "ml",
            "disease": record.disease,
            "risk_level": record.risk_level,
            "risk_score": record.risk_score,
            "created_at": record.created_at,
        }
        for record in predictions
    ]
    rows.extend(
        {
            "id": f"comparison-{record.id}",
            "record_type": "comparison",
            "disease": record.disease,
            "risk_level": record.ml_risk_level,
            "risk_score": record.ml_risk_score,
            "ml_risk_level": record.ml_risk_level,
            "ml_risk_score": record.ml_risk_score,
            "dl_risk_level": record.dl_risk_level,
            "dl_risk_score": record.dl_risk_score,
            "created_at": record.created_at,
        }
        for record in comparisons
    )
    rows.sort(key=lambda row: row["created_at"], reverse=True)
    return rows
