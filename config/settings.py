"""
config/settings.py
──────────────────
Single source of truth for every configurable value.
All values are read from environment variables (or a .env file loaded
by python-dotenv).  Hard-coded credentials from the original notebook
have been REMOVED — pass them through environment / secrets manager.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ── Load .env if present (local dev only) ────────────────────────────────────
load_dotenv()

# ── Project root ─────────────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).resolve().parent.parent
LOGS_DIR   = ROOT_DIR / os.getenv("LOG_DIR", "logs")
MODELS_DIR = ROOT_DIR / "saved_models"

LOGS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ── Environment ───────────────────────────────────────────────────────────────
ENV       = os.getenv("ENV", "development")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# ── Google Cloud Storage ──────────────────────────────────────────────────────
GCS_PROJECT_ID = os.getenv("GCS_PROJECT_ID", "united-monument-388200")
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "face-emotion-dataset")
GCS_KEY_PATH   = os.getenv("GCS_KEY_PATH", "secrets/gcs_key.json")

# Remote paths inside the bucket
GCS_TEAM_ZIP          = "TeamFaces.zip"
GCS_RAFDB_ZIP         = "raf-db.zip"
GCS_MODEL_PREFIX      = "saved_model"

# ── Local data paths ──────────────────────────────────────────────────────────
DATA_DIR          = ROOT_DIR / "data"
RAW_DIR           = DATA_DIR / "raw"
PROCESSED_DIR     = DATA_DIR / "processed"
TEAM_FACES_DIR    = RAW_DIR  / "TeamFaces"
RAFDB_DIR         = RAW_DIR  / "raf-db"

for _d in [RAW_DIR, PROCESSED_DIR, TEAM_FACES_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── Model artefact file names ─────────────────────────────────────────────────
MODEL_FILES = {
    "face_model"      : MODELS_DIR / "team_face_model.pkl",
    "label_encoder"   : MODELS_DIR / "team_label_encoder.pkl",
    "mean_embeddings" : MODELS_DIR / "team_mean_embeddings.pkl",
    "config"          : MODELS_DIR / "team_config.json",
    "embed_cache"     : MODELS_DIR / "embeddings_cache.pkl",
}

# ── Training hyperparameters ──────────────────────────────────────────────────
MIN_PHOTOS_PER_PERSON  = 5
SVM_C                  = 10.0
SVM_KERNEL             = "rbf"
TRAIN_TEST_SPLIT       = 0.2
AUGMENT_NOISE_STD      = 0.01
RANDOM_STATE           = 42

# ── Identity / recognition thresholds ────────────────────────────────────────
SVM_CONFIDENCE_THRESHOLD    = float(os.getenv("SVM_CONFIDENCE_THRESHOLD",  "0.60"))
COSINE_SIMILARITY_THRESHOLD = float(os.getenv("COSINE_SIMILARITY_THRESHOLD","0.40"))

# ── Emotion inference thresholds ─────────────────────────────────────────────
EMOTION_CONFIDENCE_THRESHOLD = float(os.getenv("EMOTION_CONFIDENCE_THRESHOLD", "55.0"))
VOTE_BUFFER_SIZE             = 5
COOLDOWN_SECONDS             = 0.4
HARD_SWITCH_THRESHOLD        = 75.0

TRACKED_EMOTIONS = ["happy", "neutral", "surprise", "angry", "sad", "fear", "disgust"]

EMOTION_GROUPS = {
    "angry"   : ["angry", "disgust"],
    "sad"     : ["sad"],
    "happy"   : ["happy"],
    "neutral" : ["neutral"],
    "surprise": ["surprise", "fear"],
}

# ── Colour maps ───────────────────────────────────────────────────────────────
EMOTION_COLORS_BGR = {
    "happy"   : (0, 255, 0),
    "neutral" : (255, 255, 0),
    "surprise": (0, 165, 255),
    "angry"   : (0, 0, 255),
    "sad"     : (255, 100, 50),
    "fear"    : (128, 0, 128),
    "disgust" : (0, 128, 0),
}

EMOTION_BOX_COLORS_RGB = {
    "angry"   : (244,  67,  54),
    "happy"   : ( 76, 175,  80),
    "neutral" : (255, 152,   0),
    "sad"     : (255, 193,   7),
    "surprise": (156,  39, 176),
    "unknown" : (255, 255, 255),
}

EMOTION_HEX = {
    "angry"   : "#F44336",
    "happy"   : "#4CAF50",
    "neutral" : "#FF9800",
    "sad"     : "#FFC107",
    "surprise": "#9C27B0",
    "unknown" : "#FFFFFF",
}

# ── DeepFace / OpenCV ─────────────────────────────────────────────────────────
DEEPFACE_MODEL         = "Facenet"
FACE_DETECTOR_BACKENDS = ["opencv", "mtcnn", "retinaface"]  # tried in order
MAX_IMAGE_WIDTH        = 640    # resize wider images before embedding
JPEG_QUALITY           = 92

# ── API ───────────────────────────────────────────────────────────────────────
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
default_api_workers = "1" if os.name == "nt" else "2"
API_WORKERS = int(os.getenv("API_WORKERS", default_api_workers))

# ── Security ──────────────────────────────────────────────────────────────────
API_KEY             = os.getenv("API_KEY", "change_me_in_production")
MAX_UPLOAD_SIZE_MB  = int(os.getenv("MAX_UPLOAD_SIZE_MB", "5"))
MAX_RUNTIME_MINUTES = int(os.getenv("MAX_RUNTIME_MINUTES", "10"))
RATE_LIMIT          = os.getenv("RATE_LIMIT", "10/minute")
LOW_LIGHT_THRESHOLD = int(os.getenv("LOW_LIGHT_THRESHOLD", "40"))
