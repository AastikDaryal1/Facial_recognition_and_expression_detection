"""
pipelines/inference.py
───────────────────────
PIPELINE 5 — INFERENCE  (Static Images / Group Photos)

Responsibility
--------------
Given an input image:
1. Detect all faces using multi-backend cascade.
2. For each face:
   a. Extract FaceNet embedding → run FaceRecognizer (SVM + cosine gate).
   b. Run detect_with_calibration() for emotion.
3. Annotate the image with bounding boxes, names, and emotion labels.
4. Save annotated output.
5. Return structured results dict.

Usage
-----
    python -m pipelines.inference --image path/to/group.jpg
    from pipelines.inference import run
"""

from __future__ import annotations

import base64
import json
import tempfile
import time
from pathlib import Path
from typing import Optional

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from deepface import DeepFace

from config.settings import (
    EMOTION_BOX_COLORS_RGB,
    EMOTION_HEX,
    FACE_DETECTOR_BACKENDS,
    LOW_LIGHT_THRESHOLD,
)
from models.emotion_model import detect_with_calibration
from models.face_model import FaceRecognizer
from utils.image_utils import bgr_to_rgb, load_image, save_image
from utils.logger import get_logger

log = get_logger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
FONT          = cv2.FONT_HERSHEY_SIMPLEX
BOX_THICKNESS = 3
LEGEND_H      = 80
PAD           = 10
MIN_FACE_SIZE = 20


def _detect_faces(img_bgr: np.ndarray, img_path: str) -> list:
    """Try detector backends in priority order; return first non-empty result."""
    for backend in FACE_DETECTOR_BACKENDS:
        try:
            faces = DeepFace.extract_faces(
                img_path=img_path,
                detector_backend=backend,
                enforce_detection=False,
                align=True,
            )
            if faces:
                log.info("  Face detector: %-12s → %d face(s)", backend, len(faces))
                return faces
        except Exception as exc:
            log.warning("  %s failed: %s", backend, exc)
    return []


def _get_embedding(face_bgr: np.ndarray) -> Optional[np.ndarray]:
    """Extract FaceNet embedding from a cropped face BGR array."""
    try:
        result = DeepFace.represent(
            img_path=face_bgr,
            model_name="Facenet",
            enforce_detection=False,
            detector_backend="skip",
        )
        return np.array(result[0]["embedding"], dtype=np.float32) if result else None
    except Exception as exc:
        log.debug("Embedding error: %s", exc)
        return None


def _annotate_image(
    canvas: np.ndarray,
    results: list[dict],
    h_orig: int,
    w_orig: int,
) -> np.ndarray:
    """Draw bounding boxes, labels, and emotion legend onto the canvas."""
    for r in results:
        x, y, w, h = r["x"], r["y"], r["w"], r["h"]
        rgb    = EMOTION_BOX_COLORS_RGB.get(r["emotion"], EMOTION_BOX_COLORS_RGB["unknown"])
        color  = (rgb[2], rgb[1], rgb[0])  # BGR

        # Face box
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, BOX_THICKNESS)

        # Label background
        label_h = 52
        cv2.rectangle(canvas, (x, max(0, y - label_h)), (x + w, y), color, -1)

        # Name text
        name_text = f"{r['name']} ({r['name_conf']:.0f}%)" if r["name"] != "UNKNOWN" else "UNKNOWN"
        cv2.putText(canvas, name_text,
                    (x + 4, max(label_h - 30, y - 30)),
                    FONT, 0.55, (0, 0, 0), 2, cv2.LINE_AA)

        # Emotion text
        emo_text = f"{r['emotion'].upper()} ({r['emotion_conf']:.0f}%)"
        cv2.putText(canvas, emo_text,
                    (x + 4, max(label_h - 10, y - 10)),
                    FONT, 0.48, (0, 0, 0), 1, cv2.LINE_AA)

    # Legend
    legend_y, legend_x = h_orig + 15, 20
    canvas[h_orig:] = (18, 18, 18)
    cv2.putText(canvas, "Legend:", (legend_x, legend_y + 16),
                FONT, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    legend_x += 75
    for label_txt, emo_key in [
        ("Angry", "angry"), ("Happy", "happy"),
        ("Neutral", "neutral"), ("Sad", "sad"), ("Surprise", "surprise"),
    ]:
        c   = EMOTION_BOX_COLORS_RGB[emo_key]
        bgr = (c[2], c[1], c[0])
        cv2.rectangle(canvas, (legend_x, legend_y), (legend_x + 28, legend_y + 22), bgr, -1)
        cv2.putText(canvas, label_txt, (legend_x + 33, legend_y + 16),
                    FONT, 0.48, (220, 220, 220), 1, cv2.LINE_AA)
        legend_x += 130

    # Summary info bar
    n_id  = sum(1 for r in results if r["name"] != "UNKNOWN")
    info  = (f"Faces: {len(results)}  |  Identified: {n_id}  |  "
             f"Unknown: {len(results) - n_id}")
    cv2.putText(canvas, info, (20, h_orig + LEGEND_H - 12),
                FONT, 0.45, (150, 150, 150), 1, cv2.LINE_AA)

    return canvas


def run(
    image_path     : str | Path,
    output_dir     : str | Path = Path("data/output"),
    recognizer     : Optional[FaceRecognizer] = None,
    save_annotated : bool = True,
) -> dict:
    """
    Execute the inference pipeline on a single image.

    Parameters
    ----------
    image_path     : input image (jpg / png / heic)
    output_dir     : where to save annotated output
    recognizer     : pre-loaded FaceRecognizer (lazily created if None)
    save_annotated : whether to write output images to disk

    Returns
    -------
    dict with keys:
        results         – list[dict]  per-face results
        n_faces         – int
        n_identified    – int
        output_path     – str | None
        elapsed_s       – float
    """
    t0         = time.time()
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("PIPELINE 5 — INFERENCE  (%s)", image_path.name)
    log.info("=" * 60)

    from config.settings import MAX_IMAGE_WIDTH
    # ── Step 1: Load image ────────────────────────────────────────────────
    img_bgr = load_image(image_path, max_width=MAX_IMAGE_WIDTH)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")
    h_orig, w_orig = img_bgr.shape[:2]
    log.info("Image: %d × %d px", w_orig, h_orig)

    # ── Step 2: Load recognizer ───────────────────────────────────────────
    if recognizer is None:
        if not FaceRecognizer.is_available():
            raise RuntimeError("Model artefacts not found. Run training pipeline first.")
        recognizer = FaceRecognizer()

    # ── Step 3: Detect faces ──────────────────────────────────────────────
    log.info("Step 1: Detecting faces …")
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_img_path = tmp.name
    cv2.imwrite(tmp_img_path, img_bgr)
    try:
        face_objs = _detect_faces(img_bgr, tmp_img_path)
    finally:
        Path(tmp_img_path).unlink(missing_ok=True)

    log.info("  %d face(s) detected.", len(face_objs))

    # ── Step 4: Per-face identity + emotion ──────────────────────────────
    log.info("Step 2: Identifying + emotion per face …")
    results: list[dict] = []

    for i, face_obj in enumerate(face_objs):
        region = face_obj.get("facial_area", face_obj.get("region", {}))
        x, y   = region.get("x", 0), region.get("y", 0)
        w, h   = region.get("w", 0), region.get("h", 0)

        if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
            log.debug("  Face %d too small (%d×%d) — skipped", i + 1, w, h)
            continue

        x1 = max(0, x - PAD);  y1 = max(0, y - PAD)
        x2 = min(w_orig, x + w + PAD);  y2 = min(h_orig, y + h + PAD)
        face_crop = img_bgr[y1:y2, x1:x2]

        # Identity
        identity = {"name": "UNKNOWN", "confidence": 0.0}
        
        # Check brightness to mitigate low-light hallucination
        gray_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray_face)

        if brightness < LOW_LIGHT_THRESHOLD:
            log.warning("  Face %d is too dark (brightness=%.1f < %d). Forcing UNKNOWN.", i+1, brightness, LOW_LIGHT_THRESHOLD)
        else:
            emb = _get_embedding(face_crop)
            if emb is not None:
                identity = recognizer.predict(emb)

        # Emotion
        emotion, emotion_conf = "neutral", 0.0
        try:
            dominant, scores = detect_with_calibration(face_crop)
            if dominant:
                emotion      = dominant
                emotion_conf = scores.get(dominant, 0.0)
        except Exception as e:
            log.warning("Emotion error: %s", e)

        face_image_b64 = None
        try:
            face_resized = cv2.resize(face_crop, (100, 100))
            _, buffer = cv2.imencode(".jpg", face_resized, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            face_image_b64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")
        except Exception as e:
            log.warning("Failed to encode face crop: %s", e)

        r = {
            "face_idx"   : i,
            "name"       : identity["name"],
            "name_conf"  : identity["confidence"],
            "emotion"    : emotion,
            "emotion_conf": emotion_conf,
            "x": x, "y": y, "w": w, "h": h,
            "face_image" : face_image_b64,
        }
        results.append(r)
        log.info(
            "  Face %d → %-20s (%.0f%%) | %-10s (%.0f%%)",
            i + 1, r["name"], r["name_conf"], r["emotion"], r["emotion_conf"],
        )

    # ── Step 5: Annotate + save ───────────────────────────────────────────
    output_path = None
    if save_annotated:
        log.info("Step 3: Annotating …")
        canvas = np.zeros((h_orig + LEGEND_H, w_orig, 3), dtype=np.uint8)
        canvas[:h_orig] = img_bgr.copy()
        canvas = _annotate_image(canvas, results, h_orig, w_orig)

        out_jpg = output_dir / (image_path.stem + "_detected.jpg")
        save_image(canvas, out_jpg)
        output_path = str(out_jpg)
        log.info("  Saved: %s", out_jpg)

    # ── Step 6: Save JSON results ─────────────────────────────────────────
    json_out = output_dir / (image_path.stem + "_results.json")
    with open(json_out, "w") as f:
        json.dump(results, f, indent=2)
    log.info("  Results JSON: %s", json_out)

    elapsed = time.time() - t0
    n_id    = sum(1 for r in results if r["name"] != "UNKNOWN")
    log.info("Inference complete in %.2f s — %d faces, %d identified.", elapsed, len(results), n_id)

    return {
        "results"      : results,
        "n_faces"      : len(results),
        "n_identified" : n_id,
        "output_path"  : output_path,
        "elapsed_s"    : elapsed,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-dir", default="data/output")
    args = parser.parse_args()
    run(args.image, args.output_dir)
