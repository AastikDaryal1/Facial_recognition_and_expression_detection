# """
# api/app.py
# ───────────
# FastAPI REST server — production deployment interface.

# Endpoints
# ---------
# GET  /health                  → liveness check
# GET  /model/info              → loaded model metadata
# POST /predict/image           → analyse an uploaded image file
# POST /predict/base64          → analyse a base64-encoded image
# GET  /metrics                 → basic request/latency metrics

# Usage
# -----
#     uvicorn api.app:app --host 0.0.0.0 --port 8000 --workers 2
#     python -m api.app
# """


# import base64
# import io
# import time
# import tempfile
# from contextlib import asynccontextmanager
# from pathlib import Path
# from typing import Optional

# import cv2
# import numpy as np
# from fastapi import FastAPI, File, HTTPException, UploadFile, status, Security, Depends, Request
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import JSONResponse
# from fastapi.security import APIKeyHeader
# from pydantic import BaseModel
# from slowapi import Limiter, _rate_limit_exceeded_handler
# from slowapi.util import get_remote_address
# from slowapi.errors import RateLimitExceeded

# from config.settings import API_HOST, API_PORT, API_WORKERS, API_KEY, MAX_UPLOAD_SIZE_MB, RATE_LIMIT
# from models.face_model import FaceRecognizer
# from pipelines.inference import run as inference_run
# from utils.logger import get_logger

# log = get_logger(__name__)

# # ─────────────────────────────────────────────────────────────────────────────
# # Shared state (loaded once at startup)
# # ─────────────────────────────────────────────────────────────────────────────
# _state: dict = {
#     "recognizer"    : None,
#     "request_count" : 0,
#     "total_latency" : 0.0,
#     "startup_time"  : None,
# }


# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     """Load model artefacts once on startup; release on shutdown."""
#     log.info("API startup — loading model artefacts …")
#     _state["startup_time"] = time.time()
#     if FaceRecognizer.is_available():
#         _state["recognizer"] = FaceRecognizer()
#         log.info("FaceRecognizer loaded ✅")
#     else:
#         log.warning("Model artefacts not found — /predict endpoints will return 503.")
#     yield
#     log.info("API shutdown.")


# # ─────────────────────────────────────────────────────────────────────────────
# # App
# # ─────────────────────────────────────────────────────────────────────────────
# limiter = Limiter(key_func=get_remote_address)

# _api_key_header = APIKeyHeader(name="X-API-Key")

# def get_api_key(api_key_header: str = Security(_api_key_header)):
#     if api_key_header != API_KEY:
#         raise HTTPException(
#             status_code=status.HTTP_401_UNAUTHORIZED,
#             detail="Invalid API Key",
#         )
#     return api_key_header

# app = FastAPI(
#     title       = "Face & Emotion Detection API",
#     description = "Identifies team members and detects emotions in photos.",
#     version     = "1.0.0",
#     lifespan    = lifespan,
# )

# app.state.limiter = limiter
# app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins  = ["*"],
#     allow_methods  = ["*"],
#     allow_headers  = ["*"],
# )


# # ─────────────────────────────────────────────────────────────────────────────
# # Pydantic models
# # ─────────────────────────────────────────────────────────────────────────────
# class FaceResult(BaseModel):
#     face_idx    : int
#     name        : str
#     name_conf   : float
#     emotion     : str
#     emotion_conf: float
#     x: int; y: int; w: int; h: int
#     face_image  : Optional[str] = None


# class PredictResponse(BaseModel):
#     n_faces         : int
#     n_identified    : int
#     results         : list[FaceResult]
#     elapsed_s       : float
#     annotated_image : Optional[str] = None  # Base64 encoded JPEG


# class Base64Request(BaseModel):
#     image_b64   : str          # base64-encoded JPEG/PNG
#     filename    : str = "input.jpg"


# class HealthResponse(BaseModel):
#     status      : str
#     model_loaded: bool
#     uptime_s    : float


# class MetricsResponse(BaseModel):
#     request_count : int
#     avg_latency_s : float
#     uptime_s      : float


# # ─────────────────────────────────────────────────────────────────────────────
# # Helper
# # ─────────────────────────────────────────────────────────────────────────────
# def _require_model() -> FaceRecognizer:
#     if _state["recognizer"] is None:
#         raise HTTPException(
#             status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
#             detail="Model not loaded. Run training pipeline first.",
#         )
#     return _state["recognizer"]


# def _run_inference(image_bytes: bytes, filename: str, save_annotated: bool = True) -> dict:
#     """Write bytes to a temp file, run inference, return result dict."""
#     suffix = Path(filename).suffix or ".jpg"
#     with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
#         tmp.write(image_bytes)
#         tmp_path = tmp.name
#     try:
#         return inference_run(
#             image_path    = tmp_path,
#             output_dir    = "data/output/api",
#             recognizer    = _state["recognizer"],
#             save_annotated= save_annotated,
#         )
#     finally:
#         Path(tmp_path).unlink(missing_ok=True)


# # ─────────────────────────────────────────────────────────────────────────────
# # Endpoints
# # ─────────────────────────────────────────────────────────────────────────────

# @app.get("/health", response_model=HealthResponse, tags=["System"])
# async def health():
#     """Liveness probe — always responds 200 while the process is alive."""
#     uptime = time.time() - (_state["startup_time"] or time.time())
#     return HealthResponse(
#         status       = "ok",
#         model_loaded = _state["recognizer"] is not None,
#         uptime_s     = round(uptime, 1),
#     )


# @app.get("/model/info", tags=["System"])
# @limiter.limit(RATE_LIMIT)
# async def model_info(request: Request, api_key: str = Security(get_api_key)):
#     """Return metadata about the loaded model."""
#     rec = _require_model()
#     return {
#         "members"         : rec.members,
#         "svm_threshold"   : rec.svm_threshold,
#         "cosine_threshold": rec.cosine_threshold,
#         "config"          : rec.config,
#     }


# @app.get("/metrics", response_model=MetricsResponse, tags=["System"])
# @limiter.limit(RATE_LIMIT)
# async def metrics(request: Request, api_key: str = Security(get_api_key)):
#     """Return basic request-count and average-latency metrics."""
#     count  = _state["request_count"]
#     total  = _state["total_latency"]
#     uptime = time.time() - (_state["startup_time"] or time.time())
#     return MetricsResponse(
#         request_count = count,
#         avg_latency_s = round(total / count, 3) if count else 0.0,
#         uptime_s      = round(uptime, 1),
#     )


# @app.post(
#     "/predict/image",
#     response_model=PredictResponse,
#     tags=["Prediction"],
#     summary="Analyse an uploaded image file",
# )
# @limiter.limit(RATE_LIMIT)
# async def predict_image(
#     request: Request,
#     file: UploadFile = File(...),
#     api_key: str = Security(get_api_key)
# ):
#     """
#     Upload a JPEG/PNG image.
#     Returns identity + emotion for each detected face.
#     """
#     _require_model()
#     t0 = time.time()

#     is_image   = file.content_type and file.content_type.startswith("image/")
#     is_generic = file.content_type == "application/octet-stream"

#     if not (is_image or is_generic):
#         raise HTTPException(400, f"Expected image, got: {file.content_type}")

#     image_bytes = await file.read()
#     if len(image_bytes) == 0:
#         raise HTTPException(400, "Empty file uploaded.")
#     if len(image_bytes) > MAX_UPLOAD_SIZE_MB * 1024 * 1024:
#         raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, f"File size exceeds {MAX_UPLOAD_SIZE_MB}MB limit.")

#     log.info("Processing upload: %s (%d bytes, type: %s)", file.filename, len(image_bytes), file.content_type)
#     try:
#         result = _run_inference(image_bytes, file.filename or "upload.jpg")
#     except Exception as exc:
#         log.error("Inference failed: %s", exc, exc_info=True)
#         raise HTTPException(500, f"Inference error: {exc}")

#     elapsed = time.time() - t0
#     _state["request_count"] += 1
#     _state["total_latency"]  += elapsed

#     # Encode annotated image to base64 for frontend preview
#     annotated_b64 = None
#     if result.get("output_path"):
#         try:
#             with open(result["output_path"], "rb") as f:
#                 annotated_b64 = base64.b64encode(f.read()).decode("utf-8")
#         except Exception as e:
#             log.warning("Could not encode output image: %s", e)

#     return PredictResponse(
#         n_faces         = result["n_faces"],
#         n_identified    = result["n_identified"],
#         results         = [FaceResult(**r) for r in result["results"]],
#         elapsed_s       = round(elapsed, 3),
#         annotated_image = annotated_b64,
#     )


# @app.post(
#     "/predict/base64",
#     response_model=PredictResponse,
#     tags=["Prediction"],
#     summary="Analyse a base64-encoded image",
# )
# @limiter.limit(RATE_LIMIT)
# async def predict_base64(
#     request: Request,
#     payload: Base64Request,
#     api_key: str = Security(get_api_key)
# ):
#     """
#     Send a base64-encoded image string.
#     Useful for browser/mobile clients that can't do multipart uploads.
#     """
#     _require_model()
#     t0 = time.time()

#     try:
#         image_bytes = base64.b64decode(payload.image_b64)
#     except Exception:
#         raise HTTPException(400, "Invalid base64 string.")

#     if len(image_bytes) > MAX_UPLOAD_SIZE_MB * 1024 * 1024:
#         raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, f"File size exceeds {MAX_UPLOAD_SIZE_MB}MB limit.")

#     try:
#         result = _run_inference(image_bytes, payload.filename, save_annotated=False)
#     except Exception as exc:
#         log.error("Inference failed: %s", exc, exc_info=True)
#         raise HTTPException(500, f"Inference error: {exc}")

#     elapsed = time.time() - t0
#     _state["request_count"] += 1
#     _state["total_latency"]  += elapsed

#     return PredictResponse(
#         n_faces      = result["n_faces"],
#         n_identified = result["n_identified"],
#         results      = [FaceResult(**r) for r in result["results"]],
#         elapsed_s    = round(elapsed, 3),
#     )


# # ─────────────────────────────────────────────────────────────────────────────
# # Dev entry point
# # ─────────────────────────────────────────────────────────────────────────────
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(
#         "api.app:app",
#         host    = API_HOST,
#         port    = API_PORT,
#         workers = 1,        # use 1 in dev; set via env in production
#         reload  = True,
#     )







import base64
import time
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile, Request, Depends, Form
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter
from slowapi.util import get_remote_address

from sqlalchemy.orm import Session
from api.db import SessionLocal
from api.models import User

from config.settings import MAX_UPLOAD_SIZE_MB
from models.face_model import FaceRecognizer
from pipelines.inference import run as inference_run

# JWT + RBAC
from rbac.role_checker import require_role
from auth.jwt_handler import create_access_token

import os
from dotenv import load_dotenv

# ─────────────────────────────────────────
# LOAD ENV
# ─────────────────────────────────────────
load_dotenv()

ORG_ADMIN_SECRET = os.getenv("ORG_ADMIN_SECRET")
SUPER_ADMIN_SECRET = os.getenv("SUPER_ADMIN_SECRET")


# ─────────────────────────────────────────
# DB CONNECTION
# ─────────────────────────────────────────
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ─────────────────────────────────────────
# STATE
# ─────────────────────────────────────────
_state = {
    "recognizer": None,
    "request_count": 0,
    "startup_time": None,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state["startup_time"] = time.time()
    if FaceRecognizer.is_available():
        _state["recognizer"] = FaceRecognizer()
    yield


# ─────────────────────────────────────────
# APP
# ─────────────────────────────────────────
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter


# ─────────────────────────────────────────
# AUTH APIs (DB + JWT)
# ─────────────────────────────────────────
@app.post("/signup", tags=["Auth"])
def signup(
    username: str = Form(...),
    password: str = Form(...),
    role: str = Form(...),
    admin_secret: str = Form(None),
    db: Session = Depends(get_db),
):
    # check existing user
    existing = db.query(User).filter(User.username == username).first()
    if existing:
        raise HTTPException(status_code=400, detail="User already exists")

    # 🔐 ROLE SECURITY
    if role == "org_admin":
        if admin_secret != ORG_ADMIN_SECRET:
            raise HTTPException(status_code=403, detail="Invalid org admin secret")

    elif role == "super_admin":
        if admin_secret != SUPER_ADMIN_SECRET:
            raise HTTPException(status_code=403, detail="Invalid super admin secret")

    # create user
    new_user = User(username=username, password=password, role=role)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return {"msg": "User created successfully"}


@app.post("/login", tags=["Auth"])
def login(
    username: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db),
):
    user = db.query(User).filter(User.username == username).first()

    if not user or user.password != password:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_access_token({
        "username": user.username,
        "role": user.role
    })

    return {"access_token": token}


# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────
def _require_model():
    if _state["recognizer"] is None:
        raise HTTPException(503, "Model not loaded")
    return _state["recognizer"]


def _run_inference(image_bytes: bytes):
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp.write(image_bytes)
        tmp_path = tmp.name

    try:
        return inference_run(image_path=tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ─────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model/info")
def model_info(user=Depends(require_role(["org_admin", "super_admin"]))):
    rec = _require_model()
    return {"members": rec.members}


@app.get("/metrics")
def metrics(user=Depends(require_role(["org_admin", "super_admin"]))):
    return {"requests": _state["request_count"]}


@app.post("/predict/image")
async def predict_image(
    file: UploadFile = File(...),
    user=Depends(require_role(["user", "org_admin", "super_admin"]))
):
    _require_model()

    image_bytes = await file.read()

    if len(image_bytes) == 0:
        raise HTTPException(400, "Empty file")

    if len(image_bytes) > MAX_UPLOAD_SIZE_MB * 1024 * 1024:
        raise HTTPException(413, "File too large")

    result = _run_inference(image_bytes)

    _state["request_count"] += 1

    return result


@app.post("/predict/base64")
def predict_base64(
    image_b64: str,
    user=Depends(require_role(["user", "org_admin", "super_admin"]))
):
    _require_model()

    try:
        image_bytes = base64.b64decode(image_b64)
    except:
        raise HTTPException(400, "Invalid base64")

    result = _run_inference(image_bytes)

    _state["request_count"] += 1

    return result


# ─────────────────────────────────────────
# RUN
# ─────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)