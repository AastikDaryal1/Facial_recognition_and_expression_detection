"""
api/app.py
───────────
FastAPI REST server — production deployment interface.

Endpoints
---------
GET  /health                  → liveness check
GET  /model/info              → loaded model metadata
POST /predict/image           → analyse an uploaded image file
POST /predict/base64          → analyse a base64-encoded image
GET  /metrics                 → basic request/latency metrics

Usage
-----
    uvicorn api.app:app --host 0.0.0.0 --port 8000 --workers 2
    python -m api.app
"""


import base64
import io
import time
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, status, Security, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from config.settings import API_HOST, API_PORT, API_WORKERS, API_KEY, MAX_UPLOAD_SIZE_MB, RATE_LIMIT
from models.face_model import FaceRecognizer
from pipelines.inference import run as inference_run
from utils.logger import get_logger

log = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Shared state (loaded once at startup)
# ─────────────────────────────────────────────────────────────────────────────
_state: dict = {
    "recognizer"    : None,
    "request_count" : 0,
    "total_latency" : 0.0,
    "startup_time"  : None,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artefacts once on startup; release on shutdown."""
    log.info("API startup — loading model artefacts …")
    _state["startup_time"] = time.time()
    if FaceRecognizer.is_available():
        _state["recognizer"] = FaceRecognizer()
        log.info("FaceRecognizer loaded ✅")
    else:
        log.warning("Model artefacts not found — /predict endpoints will return 503.")
    yield
    log.info("API shutdown.")


# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)

_api_key_header = APIKeyHeader(name="X-API-Key")

def get_api_key(api_key_header: str = Security(_api_key_header)):
    if api_key_header != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key",
        )
    return api_key_header

app = FastAPI(
    title       = "Face & Emotion Detection API",
    description = "Identifies team members and detects emotions in photos.",
    version     = "1.0.0",
    lifespan    = lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic models
# ─────────────────────────────────────────────────────────────────────────────
class FaceResult(BaseModel):
    face_idx    : int
    name        : str
    name_conf   : float
    emotion     : str
    emotion_conf: float
    x: int; y: int; w: int; h: int
    face_image  : Optional[str] = None


class PredictResponse(BaseModel):
    n_faces         : int
    n_identified    : int
    results         : list[FaceResult]
    elapsed_s       : float
    annotated_image : Optional[str] = None  # Base64 encoded JPEG


class Base64Request(BaseModel):
    image_b64   : str          # base64-encoded JPEG/PNG
    filename    : str = "input.jpg"


class HealthResponse(BaseModel):
    status      : str
    model_loaded: bool
    uptime_s    : float


class MetricsResponse(BaseModel):
    request_count : int
    avg_latency_s : float
    uptime_s      : float


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────
def _require_model() -> FaceRecognizer:
    if _state["recognizer"] is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Run training pipeline first.",
        )
    return _state["recognizer"]


def _run_inference(image_bytes: bytes, filename: str, save_annotated: bool = True) -> dict:
    """Write bytes to a temp file, run inference, return result dict."""
    suffix = Path(filename).suffix or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(image_bytes)
        tmp_path = tmp.name
    try:
        return inference_run(
            image_path    = tmp_path,
            output_dir    = "data/output/api",
            recognizer    = _state["recognizer"],
            save_annotated= save_annotated,
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Liveness probe — always responds 200 while the process is alive."""
    uptime = time.time() - (_state["startup_time"] or time.time())
    return HealthResponse(
        status       = "ok",
        model_loaded = _state["recognizer"] is not None,
        uptime_s     = round(uptime, 1),
    )


@app.get("/model/info", tags=["System"])
@limiter.limit(RATE_LIMIT)
async def model_info(request: Request, api_key: str = Security(get_api_key)):
    """Return metadata about the loaded model."""
    rec = _require_model()
    return {
        "members"         : rec.members,
        "svm_threshold"   : rec.svm_threshold,
        "cosine_threshold": rec.cosine_threshold,
        "config"          : rec.config,
    }


@app.get("/metrics", response_model=MetricsResponse, tags=["System"])
@limiter.limit(RATE_LIMIT)
async def metrics(request: Request, api_key: str = Security(get_api_key)):
    """Return basic request-count and average-latency metrics."""
    count  = _state["request_count"]
    total  = _state["total_latency"]
    uptime = time.time() - (_state["startup_time"] or time.time())
    return MetricsResponse(
        request_count = count,
        avg_latency_s = round(total / count, 3) if count else 0.0,
        uptime_s      = round(uptime, 1),
    )


@app.post(
    "/predict/image",
    response_model=PredictResponse,
    tags=["Prediction"],
    summary="Analyse an uploaded image file",
)
@limiter.limit(RATE_LIMIT)
async def predict_image(
    request: Request,
    file: UploadFile = File(...),
    api_key: str = Security(get_api_key)
):
    """
    Upload a JPEG/PNG image.
    Returns identity + emotion for each detected face.
    """
    _require_model()
    t0 = time.time()

    is_image   = file.content_type and file.content_type.startswith("image/")
    is_generic = file.content_type == "application/octet-stream"

    if not (is_image or is_generic):
        raise HTTPException(400, f"Expected image, got: {file.content_type}")

    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(400, "Empty file uploaded.")
    if len(image_bytes) > MAX_UPLOAD_SIZE_MB * 1024 * 1024:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, f"File size exceeds {MAX_UPLOAD_SIZE_MB}MB limit.")

    log.info("Processing upload: %s (%d bytes, type: %s)", file.filename, len(image_bytes), file.content_type)
    try:
        result = _run_inference(image_bytes, file.filename or "upload.jpg")
    except Exception as exc:
        log.error("Inference failed: %s", exc, exc_info=True)
        raise HTTPException(500, f"Inference error: {exc}")

    elapsed = time.time() - t0
    _state["request_count"] += 1
    _state["total_latency"]  += elapsed

    # Encode annotated image to base64 for frontend preview
    annotated_b64 = None
    if result.get("output_path"):
        try:
            with open(result["output_path"], "rb") as f:
                annotated_b64 = base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            log.warning("Could not encode output image: %s", e)

    return PredictResponse(
        n_faces         = result["n_faces"],
        n_identified    = result["n_identified"],
        results         = [FaceResult(**r) for r in result["results"]],
        elapsed_s       = round(elapsed, 3),
        annotated_image = annotated_b64,
    )


@app.post(
    "/predict/base64",
    response_model=PredictResponse,
    tags=["Prediction"],
    summary="Analyse a base64-encoded image",
)
@limiter.limit(RATE_LIMIT)
async def predict_base64(
    request: Request,
    payload: Base64Request,
    api_key: str = Security(get_api_key)
):
    """
    Send a base64-encoded image string.
    Useful for browser/mobile clients that can't do multipart uploads.
    """
    _require_model()
    t0 = time.time()

    try:
        image_bytes = base64.b64decode(payload.image_b64)
    except Exception:
        raise HTTPException(400, "Invalid base64 string.")

    if len(image_bytes) > MAX_UPLOAD_SIZE_MB * 1024 * 1024:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, f"File size exceeds {MAX_UPLOAD_SIZE_MB}MB limit.")

    try:
        result = _run_inference(image_bytes, payload.filename, save_annotated=False)
    except Exception as exc:
        log.error("Inference failed: %s", exc, exc_info=True)
        raise HTTPException(500, f"Inference error: {exc}")

    elapsed = time.time() - t0
    _state["request_count"] += 1
    _state["total_latency"]  += elapsed

    return PredictResponse(
        n_faces      = result["n_faces"],
        n_identified = result["n_identified"],
        results      = [FaceResult(**r) for r in result["results"]],
        elapsed_s    = round(elapsed, 3),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Dev entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.app:app",
        host    = API_HOST,
        port    = API_PORT,
        workers = 1,        # use 1 in dev; set via env in production
        reload  = True,
    )
