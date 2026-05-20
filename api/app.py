"""
api/app.py
───────────
FastAPI REST server — production deployment interface.

Endpoints
---------
GET  /health                  → liveness check (public)
GET  /model/info              → loaded model metadata (any logged-in user)
POST /predict/image           → analyse an uploaded image file (any logged-in user)
POST /predict/base64          → analyse a base64-encoded image (any logged-in user)
GET  /metrics                 → basic request/latency metrics (super_admin only)

Auth
----
POST /auth/signup             → create first super_admin account (one-time)
POST /auth/login              → returns access + refresh tokens
POST /auth/refresh            → rotate access token
POST /auth/logout             → blacklists current token
POST /auth/invite             → org_admin / super_admin issues an invite token
POST /auth/signup-invite      → register using an invite token

Usage
-----
    uvicorn api.app:app --host 0.0.0.0 --port 8000 --workers 2
    python -m api.app
"""


import base64
import time
import datetime
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile, status, Depends, Request, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from config.settings import API_HOST, API_PORT, API_WORKERS, MAX_UPLOAD_SIZE_MB, RATE_LIMIT
from models.face_model import FaceRecognizer
from pipelines.inference import run as inference_run
from utils.logger import get_logger
from api.auth.router import router as auth_router
from api.dependencies import get_current_user, require_role
from api.models import User, Session, Organisation, Person, AuditLog  # noqa: F401 — all must be imported so Base.metadata.create_all creates their tables
from db.base import engine, Base, get_db

log = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Shared state (loaded once at startup)
# ─────────────────────────────────────────────────────────────────────────────
_state: dict = {
    "recognizer"    : None,
    "request_count" : 0,
    "total_latency": 0.0,
    "startup_time": None,
    "last_gcs_sync": None,
    "gcs_watcher_status": "Idle", # Idle or Syncing
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create DB tables, load model artefacts once on startup; start GCS watcher."""

    # ── Database ──────────────────────────────────────────────────────────────
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    log.info("Database tables verified ✅")

    # ── Model ─────────────────────────────────────────────────────────────────
    log.info("API startup — loading model artefacts …")
    _state["startup_time"] = time.time()
    _load_model()

    # ── GCS Watcher ───────────────────────────────────────────────────────────
    import asyncio
    from config.settings import GCS_TEAM_ZIP
    from storage.gcs_storage import GCSStorage
    from pipelines import data_ingestion, preprocessing, feature_extraction, training

    async def gcs_watcher():
        """Background loop to check for dataset updates on GCS."""
        await asyncio.sleep(10) # wait for startup to settle
        gcs = GCSStorage()
        
        # Monitor the folder 'team_faces/'
        prefix = "team_faces/"
        last_updated = gcs.get_latest_blob_updated(prefix)
        log.info("GCS Watcher started. Monitoring folder: %s (initial: %s)", prefix, last_updated)

        while True:
            try:
                await asyncio.sleep(60) # Poll every 1 minute
                current_updated = gcs.get_latest_blob_updated(prefix)
                
                if current_updated and (last_updated is None or current_updated > last_updated):
                    _state["gcs_watcher_status"] = "Syncing"
                    log.info("🔔 GCS CHANGE DETECTED! Dataset folder '%s' updated at %s. Starting auto-sync...", prefix, current_updated)
                    
                    try:
                        # Run full pipeline
                        data_ingestion.run(force_retrain=True)
                        preprocessing.run()
                        fe = feature_extraction.run()
                        training.run(fe["X"], fe["y"], fe["X_raw"], fe["y_raw"])
                        
                        # Hot-reload the model
                        _load_model()
                        last_updated = current_updated
                        _state["last_gcs_sync"] = datetime.datetime.now().isoformat()
                        log.info("✅ Auto-sync & re-training complete. Model hot-reloaded.")
                    finally:
                        _state["gcs_watcher_status"] = "Idle"
            except Exception as e:
                log.error("GCS Watcher error: %s", e)

    watcher_task = asyncio.create_task(gcs_watcher())
    
    yield
    
    watcher_task.cancel()
    log.info("API shutdown.")


def hot_reload_model():
    """Trigger a re-load of the model from disk into the running process."""
    _load_model()


def _load_model():
    """Helper to (re)load the recognizer into global state."""
    if FaceRecognizer.is_available():
        _state["recognizer"] = FaceRecognizer()
        log.info("FaceRecognizer (re)loaded ✅")
    else:
        log.warning("Model artefacts not found — /predict endpoints will return 503.")


# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)

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
    allow_origins = ["http://localhost:5173", "http://localhost:3000", "http://192.168.19.64:3000", "http://192.168.19.69:3000", "http://127.0.0.1:3000"],
    allow_credentials = True,
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

# ── Auth router ──────────────────────────────────────────────────────────────
app.include_router(auth_router, prefix="/auth", tags=["Auth"])
from api.routers.users import router as users_router
app.include_router(users_router, prefix="/users", tags=["Users"])
from api.routers.organisations import router as orgs_router
app.include_router(orgs_router, prefix="/organisations", tags=["Organisations"])
from api.routers.persons import router as persons_router
app.include_router(persons_router, prefix="/persons", tags=["Persons"])
from api.routers.sessions import router as sessions_router
app.include_router(sessions_router, prefix="/sessions", tags=["Sessions"])
from api.routers.audit import router as audit_router
app.include_router(audit_router, prefix="/audit", tags=["Audit"])
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
    request_count      : int
    avg_latency_s      : float
    uptime_s           : float
    last_gcs_sync      : Optional[str] = None
    gcs_watcher_status : str = "Idle"


def _encode_annotated_image(result: dict) -> Optional[str]:
    """
    Return a base64-encoded JPEG of the annotated image, or None.
    For /predict/image  → reads from the saved output file (output_path).
    For /predict/base64 → output_path is None (save_annotated=False),
                          so we return None here; the live feed doesn't
                          need a persisted annotated frame.
    """
    output_path = result.get("output_path")
    if not output_path:
        return None
    try:
        with open(output_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        log.warning("Could not encode output image: %s", e)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _require_model() -> FaceRecognizer:
    if _state["recognizer"] is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Run training pipeline first.",
        )
    return _state["recognizer"]


def _run_inference(
    image_bytes: bytes,
    filename: str,
    save_annotated: bool = True,
    generate_crops: bool = True,
    detector_backends: Optional[list[str]] = None,
    is_live: bool = False,
) -> dict:
    """Write bytes to a temp file, run inference, return result dict."""
    suffix = Path(filename).suffix or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(image_bytes)
        tmp_path = tmp.name
    try:
        return inference_run(
            image_path        = tmp_path,
            output_dir        = "data/output/api",
            recognizer        = _state["recognizer"],
            save_annotated    = save_annotated,
            generate_crops    = generate_crops,
            detector_backends = detector_backends,
            is_live           = is_live,
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.api_route("/health", methods=["GET", "HEAD"], response_model=HealthResponse, tags=["System"])
async def health():
    """Liveness probe — always responds 200 while the process is alive. No auth required."""
    uptime = time.time() - (_state["startup_time"] or time.time())
    return HealthResponse(
        status       = "ok",
        model_loaded = _state["recognizer"] is not None,
        uptime_s     = round(uptime, 1),
    )


@app.get("/model/info", tags=["System"])
@limiter.limit(RATE_LIMIT)
async def model_info(
    request      : Request,
    current_user : User = Depends(get_current_user),   # any logged-in user
):
    """Return metadata about the loaded model."""
    rec = _require_model()
    return {
        "members"         : rec.members,
        "svm_threshold"   : rec.svm_threshold,
        "cosine_threshold": rec.cosine_threshold,
        "config"          : rec.config,
    }


@app.get("/metrics", response_model=MetricsResponse, tags=["System"])
@limiter.limit("120/minute")  # dashboard polls this frequently; higher limit than public endpoints
async def metrics(
    request      : Request,
    current_user : User = Depends(require_role(["super_admin"])),  # super_admin only
):
    return MetricsResponse(
        request_count      = _state["request_count"],
        avg_latency_s      = round(_state["total_latency"] / _state["request_count"], 3) if _state["request_count"] > 0 else 0.0,
        uptime_s           = round(time.time() - (_state["startup_time"] or time.time()), 1),
        last_gcs_sync      = _state["last_gcs_sync"],
        gcs_watcher_status = _state["gcs_watcher_status"],
    )


@app.post("/system/sync", tags=["System"])
async def trigger_sync(
    current_user : User = Depends(require_role(["super_admin"])),
    background_tasks: BackgroundTasks = None
):
    """Manually trigger a GCS sync in the background."""
    if _state["gcs_watcher_status"] == "Syncing":
        return {"message": "Sync already in progress."}

    def run_sync_task():
        from storage.gcs_storage import GCSStorage
        from pipelines import data_ingestion, preprocessing, feature_extraction, training
        try:
            _state["gcs_watcher_status"] = "Syncing"
            # Ensure fresh download
            data_ingestion.run(force_retrain=True)
            preprocessing.run()
            fe = feature_extraction.run()
            training.run(fe["X"], fe["y"], fe["X_raw"], fe["y_raw"])
            _load_model()
            _state["last_gcs_sync"] = datetime.datetime.now().isoformat()
            log.info("✅ Manual sync complete.")
        finally:
            _state["gcs_watcher_status"] = "Idle"

    background_tasks.add_task(run_sync_task)
    return {"message": "Sync triggered successfully."}


@app.post(
    "/predict/image",
    response_model=PredictResponse,
    tags=["Prediction"],
    summary="Analyse an uploaded image file",
)
@limiter.limit(RATE_LIMIT)
async def predict_image(
    request      : Request,
    file         : UploadFile = File(...),
    current_user : User = Depends(get_current_user),   # any logged-in user
    db           : AsyncSession = Depends(get_db),
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
        # Upload page: prioritize accuracy for group photos (is_live=False)
        result = _run_inference(
            image_bytes,
            file.filename,
            detector_backends=["retinaface", "mtcnn", "opencv"],
            is_live=False,
        )
    except Exception as exc:
        log.error("Inference failed: %s", exc, exc_info=True)
        raise HTTPException(500, f"Inference error: {exc}")

    elapsed = time.time() - t0
    _state["request_count"] += 1
    _state["total_latency"]  += elapsed

    annotated_b64 = _encode_annotated_image(result)

    # Save session to DB
    try:
        session_rec = Session(
            user_id         = current_user.id,
            org_id          = current_user.org_id,
            n_faces         = result["n_faces"],
            n_identified    = result["n_identified"],
            elapsed_s       = round(elapsed, 3),
            results_json    = {"results": result["results"]},
            annotated_image = annotated_b64,
        )
        db.add(session_rec)
        await db.commit()
        log.info("Session recorded: %s", session_rec.id)
    except Exception as e:
        log.error("Failed to save session record: %s", e)
        # We don't raise here — inference was successful, session saving is secondary

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
    request      : Request,
    payload      : Base64Request,
    current_user : User = Depends(get_current_user),   # any logged-in user
    db           : AsyncSession = Depends(get_db),
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
        # Live mode: Enable crop generation for better visual feedback (is_live=True)
        result = _run_inference(
            image_bytes,
            payload.filename,
            save_annotated=False,
            generate_crops=True,
            is_live=True,
        )
    except Exception as exc:
        log.error("Inference failed: %s", exc, exc_info=True)
        raise HTTPException(500, f"Inference error: {exc}")

    elapsed = time.time() - t0
    _state["request_count"] += 1
    _state["total_latency"]  += elapsed

    # save_annotated=False for live frames → output_path is None → annotated_b64 will be None
    annotated_b64 = _encode_annotated_image(result)

    # Save session to DB
    try:
        session_rec = Session(
            user_id         = current_user.id,
            org_id          = current_user.org_id,
            n_faces         = result["n_faces"],
            n_identified    = result["n_identified"],
            elapsed_s       = round(elapsed, 3),
            results_json    = {"results": result["results"]},
            annotated_image = annotated_b64,
        )
        db.add(session_rec)
        await db.commit()
        log.info("Session recorded (base64): %s", session_rec.id)
    except Exception as e:
        log.error("Failed to save session record: %s", e)

    return PredictResponse(
        n_faces         = result["n_faces"],
        n_identified    = result["n_identified"],
        results         = [FaceResult(**r) for r in result["results"]],
        elapsed_s       = round(elapsed, 3),
        annotated_image = annotated_b64,
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