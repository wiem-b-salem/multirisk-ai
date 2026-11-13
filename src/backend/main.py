import os
from datetime import datetime, timedelta, timezone

import bcrypt
import httpx
import jwt
from fastapi import Body, Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field
from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, create_engine, select
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


def validate_settings(environment: str, jwt_secret: str) -> None:
    if environment == "production" and (
        jwt_secret == "dev-secret-change-me" or len(jwt_secret) < 32
    ):
        raise RuntimeError(
            "JWT_SECRET must be a non-default value of at least 32 characters "
            "when ENVIRONMENT=production."
        )


validate_settings(ENVIRONMENT, JWT_SECRET)

if JWT_SECRET == "dev-secret-change-me":
    print("WARNING: using default JWT_SECRET; set JWT_SECRET in production.")

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


@app.get("/history")
def history(user: User = Depends(get_current_user), db=Depends(get_db)):
    records = db.scalars(
        select(PredictionRecord)
        .where(PredictionRecord.user_id == user.id)
        .order_by(PredictionRecord.created_at.desc(), PredictionRecord.id.desc())
    ).all()
    return [
        {
            "id": record.id,
            "disease": record.disease,
            "risk_level": record.risk_level,
            "risk_score": record.risk_score,
            "created_at": record.created_at,
        }
        for record in records
    ]
