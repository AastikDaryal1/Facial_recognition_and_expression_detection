"""
pipelines/data_ingestion.py
────────────────────────────
PIPELINE 1 — DATA INGESTION

Responsibility
--------------
Download raw datasets and pre-trained model artefacts from GCS.
Raises early if data is already present (idempotent).

Steps
-----
1. Check if saved model artefacts exist in the bucket.
   → If YES: download them and skip all data download steps.
   → If NO : download TeamFaces.zip (team identity images)
             download raf-db.zip  (emotion training images)
             extract both archives to data/raw/

Usage
-----
    python -m pipelines.data_ingestion          # CLI
    from pipelines.data_ingestion import run    # API / orchestrator
"""

from __future__ import annotations

import time
from pathlib import Path

from config.settings import (
    DATA_DIR,
    GCS_MODEL_PREFIX,
    GCS_RAFDB_ZIP,
    GCS_TEAM_ZIP,
    MODEL_FILES,
    RAW_DIR,
    RAFDB_DIR,
    TEAM_FACES_DIR,
)
from storage.gcs_storage import GCSStorage
from utils.logger import get_logger

log = get_logger(__name__)

# Remote model blob paths (inside the bucket)
_REMOTE_MODEL_FILES: dict[str, Path] = {
    f"{GCS_MODEL_PREFIX}/team_mean_embeddings.pkl": MODEL_FILES["mean_embeddings"],
    f"{GCS_MODEL_PREFIX}/team_label_encoder.pkl"  : MODEL_FILES["label_encoder"],
    f"{GCS_MODEL_PREFIX}/team_face_model.pkl"     : MODEL_FILES["face_model"],
    f"{GCS_MODEL_PREFIX}/team_config.json"        : MODEL_FILES["config"],
    f"{GCS_MODEL_PREFIX}/embeddings_cache.pkl"    : MODEL_FILES["embed_cache"],
}


def run(force_retrain: bool = False) -> dict:
    """
    Execute the data-ingestion pipeline.

    Parameters
    ----------
    force_retrain : bool
        If True, always download raw data even if a saved model exists.

    Returns
    -------
    dict with keys:
        model_loaded  – bool   : saved model was found and downloaded
        data_ready    – bool   : raw data extracted and ready for training
        elapsed_s     – float  : wall-clock seconds
    """
    t0 = time.time()
    log.info("=" * 60)
    log.info("PIPELINE 1 — DATA INGESTION  (force_retrain=%s)", force_retrain)
    log.info("=" * 60)

    gcs = GCSStorage()
    result = {"model_loaded": False, "data_ready": False, "elapsed_s": 0.0}

    # ── Step 1: Try to load existing saved model ──────────────────────────
    if not force_retrain:
        log.info("Step 1: Checking for saved model artefacts in bucket …")
        remote_paths = list(_REMOTE_MODEL_FILES.keys())
        if gcs.all_blobs_exist(remote_paths):
            log.info("  ✅ Saved model found!  Downloading artefacts …")
            for remote, local in _REMOTE_MODEL_FILES.items():
                gcs.download_file(remote, local)
            result["model_loaded"] = True
            result["elapsed_s"]    = time.time() - t0
            log.info("Data ingestion complete — model loaded, no training needed.")
            return result
        else:
            log.info("  ⚠️  No saved model found — will download raw data and train.")
    else:
        log.info("Step 1: force_retrain=True — skipping saved-model check.")

    # ── Step 2: Download + extract TeamFaces ─────────────────────────────
    log.info("Step 2: Downloading TeamFaces dataset …")
    team_zip = RAW_DIR / "TeamFaces.zip"
    if not team_zip.exists():
        gcs.download_and_extract_zip(GCS_TEAM_ZIP, team_zip, RAW_DIR)
    else:
        log.info("  TeamFaces.zip already present, skipping download.")
        if not TEAM_FACES_DIR.exists():
            import zipfile
            log.info("  Extracting existing zip …")
            with zipfile.ZipFile(str(team_zip)) as zf:
                zf.extractall(str(RAW_DIR))

    # ── Step 3: Download + extract raf-db ────────────────────────────────
    log.info("Step 3: Downloading raf-db dataset …")
    rafdb_zip = RAW_DIR / "raf-db.zip"
    if not rafdb_zip.exists():
        gcs.download_and_extract_zip(GCS_RAFDB_ZIP, rafdb_zip, RAW_DIR)
    else:
        log.info("  raf-db.zip already present, skipping download.")

    # ── Step 4: Validate ─────────────────────────────────────────────────
    log.info("Step 4: Validating extracted data …")
    members = [d for d in TEAM_FACES_DIR.iterdir() if d.is_dir()]
    log.info("  Team members found: %d", len(members))
    for m in members:
        imgs = list(m.glob("*.jpg")) + list(m.glob("*.jpeg")) + list(m.glob("*.png"))
        log.info("    %-20s  %d images", m.name, len(imgs))

    result["data_ready"] = True
    result["elapsed_s"]  = time.time() - t0
    log.info("Data ingestion complete in %.1f s", result["elapsed_s"])
    return result


if __name__ == "__main__":
    run()
