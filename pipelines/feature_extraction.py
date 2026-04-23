"""
pipelines/feature_extraction.py
─────────────────────────────────
PIPELINE 3 — FEATURE EXTRACTION

Responsibility
--------------
Extract 512-dimensional FaceNet embeddings for every processed image
and persist them to an on-disk cache so re-runs skip GPU work.

Steps
-----
1. Warm up the DeepFace / FaceNet model.
2. Load embeddings_cache.pkl if it already exists.
3. For each valid member → for each clean image in data/processed/:
   a. Run DeepFace.represent() with FaceNet.
   b. Store (embedding, label) pair.
4. Augment with Gaussian noise (doubles the dataset cheaply).
5. L2-normalise all embeddings.
6. Persist cache and return X, y arrays.

Usage
-----
    python -m pipelines.feature_extraction
    from pipelines.feature_extraction import run
"""

from __future__ import annotations

import pickle
import tempfile
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from deepface import DeepFace
from sklearn.preprocessing import normalize

from config.settings import (
    AUGMENT_NOISE_STD,
    DEEPFACE_MODEL,
    JPEG_QUALITY,
    MAX_IMAGE_WIDTH,
    MODEL_FILES,
    PROCESSED_DIR,
    RANDOM_STATE,
)
from utils.image_utils import is_image_file, load_image, save_image
from utils.logger import get_logger

log = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Embedding extraction
# ─────────────────────────────────────────────────────────────────────────────

def _warmup_model() -> None:
    """Run a dummy image through DeepFace so the first real call is fast."""
    log.info("Warming up FaceNet model …")
    dummy = np.ones((160, 160, 3), dtype=np.uint8) * 128
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name
    cv2.imwrite(tmp_path, dummy, [cv2.IMWRITE_JPEG_QUALITY, 92])
    try:
        DeepFace.represent(
            tmp_path,
            model_name=DEEPFACE_MODEL,
            enforce_detection=False,
            detector_backend="opencv",
        )
        log.info("FaceNet model warmed up.")
    except Exception as exc:
        log.warning("Warmup failed (non-fatal): %s", exc)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _extract_embedding(img_bgr: np.ndarray) -> Optional[np.ndarray]:
    """
    Extract a FaceNet embedding from a BGR numpy array.
    Returns a 1-D numpy array or None on failure.
    Uses a temp file on disk (DeepFace requires a file path or BGR array).
    """
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        cv2.imwrite(tmp_path, img_bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        result = DeepFace.represent(
            img_path=tmp_path,
            model_name=DEEPFACE_MODEL,
            enforce_detection=False,
            detector_backend="opencv",
        )
        if result:
            return np.array(result[0]["embedding"], dtype=np.float32)
        return None
    except Exception as exc:
        log.debug("Embedding extraction failed: %s", exc)
        return None
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _augment(embeddings: list, labels: list, noise_std: float, seed: int) -> tuple:
    """Add Gaussian noise to each embedding to artificially double dataset size."""
    rng = np.random.default_rng(seed)
    aug_embs, aug_lbls = [], []
    for emb, lbl in zip(embeddings, labels):
        noise = rng.normal(0, noise_std, size=len(emb))
        aug_embs.append(np.array(emb) + noise)
        aug_lbls.append(lbl)
    return aug_embs, aug_lbls


# ─────────────────────────────────────────────────────────────────────────────
# Public pipeline function
# ─────────────────────────────────────────────────────────────────────────────

def run(valid_members: Optional[list[str]] = None) -> dict:
    """
    Execute the feature-extraction pipeline.

    Parameters
    ----------
    valid_members : list[str] | None
        Member names to include.  If None, all sub-directories under
        PROCESSED_DIR are used.

    Returns
    -------
    dict with keys:
        X            – np.ndarray (N, 512) normalised embeddings
        y            – np.ndarray (N,)     string labels
        X_raw        – np.ndarray          original embeddings before augment
        y_raw        – np.ndarray          original labels before augment
        n_embeddings – int
        elapsed_s    – float
    """
    t0 = time.time()
    log.info("=" * 60)
    log.info("PIPELINE 3 — FEATURE EXTRACTION")
    log.info("=" * 60)

    cache_path = MODEL_FILES["embed_cache"]

    # ── Step 1: Try loading cache ─────────────────────────────────────────
    if cache_path.exists():
        log.info("Step 1: Embedding cache found at %s — loading …", cache_path)
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        embeddings = cache["embeddings"]
        labels     = cache["labels"]
        log.info("  Loaded %d cached embeddings.", len(embeddings))
    else:
        # ── Step 2: Warmup ────────────────────────────────────────────────
        log.info("Step 1: No cache found — extracting embeddings from scratch.")
        _warmup_model()

        # ── Step 3: Extract embeddings ────────────────────────────────────
        if valid_members is None:
            valid_members = [d.name for d in PROCESSED_DIR.iterdir() if d.is_dir()]

        log.info("Step 2: Extracting FaceNet embeddings for %d members …", len(valid_members))
        embeddings: list = []
        labels    : list = []
        skipped           = 0
        t_extract         = time.time()

        for member in valid_members:
            member_dir = PROCESSED_DIR / member
            if not member_dir.exists():
                log.warning("  Processed dir missing for %s — skipped", member)
                continue

            img_files = sorted([f for f in member_dir.iterdir() if is_image_file(f)])
            member_count = 0

            for idx, img_path in enumerate(img_files):
                img = load_image(img_path, max_width=MAX_IMAGE_WIDTH)
                if img is None:
                    skipped += 1
                    continue

                emb = _extract_embedding(img)
                if emb is not None:
                    embeddings.append(emb)
                    labels.append(member)
                    member_count += 1
                else:
                    skipped += 1

                if (idx + 1) % 20 == 0:
                    elapsed = time.time() - t_extract
                    done    = sum(1 for _ in labels)
                    rate    = done / max(elapsed, 1e-6)
                    log.info("    [%s] %d/%d  (%.1f img/s)", member, idx + 1, len(img_files), rate)

            log.info("  ✓ %-20s  %d embeddings", member, member_count)

        log.info(
            "  Total: %d embeddings | Skipped: %d | Time: %.1f s",
            len(embeddings), skipped, time.time() - t_extract,
        )

        if not embeddings:
            raise RuntimeError("No embeddings extracted. Check image quality / member directories.")

        # Save cache
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump({"embeddings": embeddings, "labels": labels}, f)
        log.info("  ✅ Embedding cache saved → %s", cache_path)

    # ── Step 4: Augmentation ──────────────────────────────────────────────
    log.info("Step 3: Augmenting with Gaussian noise (std=%.3f) …", AUGMENT_NOISE_STD)
    aug_embs, aug_lbls = _augment(embeddings, labels, AUGMENT_NOISE_STD, RANDOM_STATE)
    all_embs   = embeddings + aug_embs
    all_labels = labels     + aug_lbls
    log.info("  Dataset size after augmentation: %d → %d", len(embeddings), len(all_embs))

    # ── Step 5: Normalise ─────────────────────────────────────────────────
    log.info("Step 4: L2-normalising embeddings …")
    X     = normalize(np.array(all_embs, dtype=np.float32))
    y     = np.array(all_labels)
    X_raw = normalize(np.array(embeddings, dtype=np.float32))
    y_raw = np.array(labels)

    elapsed = time.time() - t0
    log.info("Feature extraction complete in %.1f s", elapsed)

    return {
        "X"           : X,
        "y"           : y,
        "X_raw"       : X_raw,
        "y_raw"       : y_raw,
        "n_embeddings": len(embeddings),
        "elapsed_s"   : elapsed,
    }


if __name__ == "__main__":
    result = run()
    log.info("X.shape=%s  Unique labels=%s", result["X"].shape, list(set(result["y"])))
