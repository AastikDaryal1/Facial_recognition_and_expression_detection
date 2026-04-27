"""
pipelines/deployment.py
────────────────────────
PIPELINE 7 — DEPLOYMENT

Responsibility
--------------
Upload trained model artefacts back to Google Cloud Storage so they
can be pulled by the data-ingestion pipeline on the next cold start
(skipping re-training).

This is intentionally kept separate from training so that you can
choose when to "promote" a newly trained model (e.g. only after it
passes offline evaluation).

Steps
-----
1. Validate that all local model files exist.
2. Upload each file to the remote bucket under saved_model/<filename>.
3. Write a deployment_manifest.json with timestamp and accuracy metrics.

Usage
-----
    python -m pipelines.deployment
    from pipelines.deployment import run
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from config.settings import GCS_MODEL_PREFIX, MODEL_FILES
from storage.gcs_storage import GCSStorage
from utils.logger import get_logger

log = get_logger(__name__)

_REMOTE_MAP: dict[str, Path] = {
    f"{GCS_MODEL_PREFIX}/team_face_model.pkl"      : MODEL_FILES["face_model"],
    f"{GCS_MODEL_PREFIX}/team_label_encoder.pkl"   : MODEL_FILES["label_encoder"],
    f"{GCS_MODEL_PREFIX}/team_mean_embeddings.pkl" : MODEL_FILES["mean_embeddings"],
    f"{GCS_MODEL_PREFIX}/team_config.json"         : MODEL_FILES["config"],
    f"{GCS_MODEL_PREFIX}/embeddings_cache.pkl"     : MODEL_FILES["embed_cache"],
}


def run(training_metrics: dict | None = None) -> dict:
    """
    Upload local model artefacts to GCS.

    Parameters
    ----------
    training_metrics : dict | None
        Optional dict returned by pipelines.training.run(), stored in the
        deployment manifest for traceability.

    Returns
    -------
    dict with keys:
        uploaded      – list[str]  remote paths uploaded
        manifest_path – str        local path of manifest file
        elapsed_s     – float
    """
    t0 = time.time()
    log.info("=" * 60)
    log.info("PIPELINE 7 — DEPLOYMENT (upload to GCS)")
    log.info("=" * 60)

    # ── Step 1: Validate local files ──────────────────────────────────────
    log.info("Step 1: Validating local model artefacts …")
    missing = [str(p) for p in MODEL_FILES.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing model files — run training first:\n  " + "\n  ".join(missing)
        )
    log.info("  All local artefacts present ✅")

    # ── Step 2: Upload ────────────────────────────────────────────────────
    log.info("Step 2: Uploading to GCS …")
    gcs      = GCSStorage()
    uploaded = []
    for remote, local in _REMOTE_MAP.items():
        gcs.upload_file(local, remote)
        uploaded.append(remote)

    # ── Step 3: Write deployment manifest ────────────────────────────────
    log.info("Step 3: Writing deployment manifest …")
    manifest = {
        "deployed_at"    : datetime.now(tz=timezone.utc).isoformat(),
        "uploaded_files" : uploaded,
        "training_metrics": training_metrics or {},
    }
    manifest_path = MODEL_FILES["config"].parent / "deployment_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    log.info("  Manifest saved → %s", manifest_path)

    elapsed = time.time() - t0
    log.info("Deployment complete in %.1f s", elapsed)
    log.info("✅ Uploaded %d files — next cold start will skip training.", len(uploaded))

    return {
        "uploaded"      : uploaded,
        "manifest_path" : str(manifest_path),
        "elapsed_s"     : elapsed,
    }


if __name__ == "__main__":
    run()
