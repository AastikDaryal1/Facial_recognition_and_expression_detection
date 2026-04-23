"""
pipelines/preprocessing.py
───────────────────────────
PIPELINE 2 — PREPROCESSING

Responsibility
--------------
Walk the raw TeamFaces directory, validate every image, and produce a
clean processed copy ready for embedding extraction.

Steps
-----
1. Scan raw TeamFaces folder — collect member directories.
2. Filter members with fewer than MIN_PHOTOS_PER_PERSON images.
3. For each valid image:
   a. Load (including HEIC conversion).
   b. Down-scale if wider than MAX_IMAGE_WIDTH.
   c. Check the image actually contains a detectable face
      (light sanity check using OpenCV Haar cascade — fast, no GPU).
   d. Write cleaned JPEG to data/processed/<member>/<file>.jpg
4. Produce a preprocessing_report.json summary.

Usage
-----
    python -m pipelines.preprocessing
    from pipelines.preprocessing import run
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from config.settings import (
    JPEG_QUALITY,
    MAX_IMAGE_WIDTH,
    MIN_PHOTOS_PER_PERSON,
    PROCESSED_DIR,
    TEAM_FACES_DIR,
)
from utils.image_utils import is_image_file, load_image, save_image
from utils.logger import get_logger

log = get_logger(__name__)

# OpenCV Haar cascade for quick face sanity-check
_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_FACE_CASCADE: Optional[cv2.CascadeClassifier] = None


def _get_cascade() -> cv2.CascadeClassifier:
    global _FACE_CASCADE
    if _FACE_CASCADE is None:
        _FACE_CASCADE = cv2.CascadeClassifier(_CASCADE_PATH)
    return _FACE_CASCADE


def _has_face(img_bgr: np.ndarray) -> bool:
    """Quick check: does image contain at least one front-on face?"""
    gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = _get_cascade().detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
    return len(faces) > 0


def _process_member(
    member_dir: Path,
    out_dir: Path,
) -> dict:
    """Process all images for one team member. Returns per-member stats."""
    stats = {"member": member_dir.name, "total": 0, "ok": 0, "no_face": 0, "failed": 0}
    out_dir.mkdir(parents=True, exist_ok=True)

    image_files = [f for f in sorted(member_dir.iterdir()) if is_image_file(f)]
    stats["total"] = len(image_files)

    for img_path in image_files:
        img = load_image(img_path, max_width=MAX_IMAGE_WIDTH)
        if img is None:
            stats["failed"] += 1
            log.warning("  Could not load %s", img_path.name)
            continue

        if not _has_face(img):
            stats["no_face"] += 1
            log.debug("  No face detected in %s — skipping", img_path.name)
            continue

        out_path = out_dir / (img_path.stem + ".jpg")
        saved    = save_image(img, out_path, jpeg_quality=JPEG_QUALITY)
        if saved:
            stats["ok"] += 1
        else:
            stats["failed"] += 1

    return stats


def run() -> dict:
    """
    Execute the preprocessing pipeline.

    Returns
    -------
    dict with keys:
        valid_members – list[str] : members with enough clean images
        report        – list[dict]: per-member stats
        elapsed_s     – float
    """
    t0 = time.time()
    log.info("=" * 60)
    log.info("PIPELINE 2 — PREPROCESSING")
    log.info("=" * 60)

    if not TEAM_FACES_DIR.exists():
        raise FileNotFoundError(
            f"Team faces directory not found: {TEAM_FACES_DIR}\n"
            "Run the data ingestion pipeline first."
        )

    # ── Step 1: Scan member directories ──────────────────────────────────
    log.info("Step 1: Scanning %s …", TEAM_FACES_DIR)
    member_dirs = sorted([d for d in TEAM_FACES_DIR.iterdir() if d.is_dir()])
    log.info("  Found %d member directories", len(member_dirs))

    # ── Step 2-3: Process each member ────────────────────────────────────
    report: list[dict] = []
    valid_members: list[str] = []

    for member_dir in member_dirs:
        log.info("  Processing member: %s", member_dir.name)
        out_dir = PROCESSED_DIR / member_dir.name
        stats   = _process_member(member_dir, out_dir)
        report.append(stats)
        log.info(
            "    Total=%d  OK=%d  No-face=%d  Failed=%d",
            stats["total"], stats["ok"], stats["no_face"], stats["failed"],
        )
        if stats["ok"] >= MIN_PHOTOS_PER_PERSON:
            valid_members.append(member_dir.name)
        else:
            log.warning(
                "    ⚠️  %s has only %d valid images (need %d) — will be skipped",
                member_dir.name, stats["ok"], MIN_PHOTOS_PER_PERSON,
            )

    # ── Step 4: Write report ──────────────────────────────────────────────
    report_path = PROCESSED_DIR / "preprocessing_report.json"
    with open(report_path, "w") as f:
        json.dump({"valid_members": valid_members, "per_member": report}, f, indent=2)
    log.info("Report saved → %s", report_path)

    elapsed = time.time() - t0
    log.info("Valid members (%d): %s", len(valid_members), valid_members)
    log.info("Preprocessing complete in %.1f s", elapsed)

    if len(valid_members) < 2:
        raise RuntimeError("Need at least 2 members with enough valid images to train.")

    return {"valid_members": valid_members, "report": report, "elapsed_s": elapsed}


if __name__ == "__main__":
    run()
