"""
utils/image_utils.py
────────────────────
Stateless helper functions for image I/O and basic transforms.
All functions accept/return numpy BGR arrays (OpenCV convention)
unless the name says otherwise.
"""

import io
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from utils.logger import get_logger

# ── Logger ───────────────────────────────────────────────────────────────────

log = get_logger(__name__)

# Supported image extensions (lower-case)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def load_image(path: str | Path, max_width: int = 0) -> Optional[np.ndarray]:
    """
    Load an image from disk.

    • Supports HEIC/HEIF via pillow-heif (auto-registered at import time).
    • Optionally down-scales images wider than `max_width`.
    Optionally down-scales images wider than `max_width`.

    Returns
    -------
    BGR numpy array or None if loading fails.
    """
    path = Path(path)
    img = cv2.imread(str(path))

    if img is None:
        log.warning("Could not load image: %s", path)
        return None

    if max_width and img.shape[1] > max_width:
        img = resize_to_width(img, max_width)

    return img


def resize_to_width(img: np.ndarray, max_width: int) -> np.ndarray:
    """Down-scale image so width ≤ max_width, keeping aspect ratio."""
    h, w = img.shape[:2]
    if w <= max_width:
        return img
    scale = max_width / w
    return cv2.resize(img, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)


def save_image(img: np.ndarray, path: str | Path, jpeg_quality: int = 92) -> bool:
    """Save BGR numpy array to disk. Returns True on success."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality] if path.suffix.lower() in {".jpg", ".jpeg"} else []
    ok = cv2.imwrite(str(path), img, params)
    if not ok:
        log.error("Failed to save image: %s", path)
    return ok


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def encode_jpeg(img: np.ndarray, quality: int = 85) -> bytes:
    """Encode BGR array → JPEG bytes (for API responses / streaming)."""
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()


def crop_face(img: np.ndarray, region: dict, pad: int = 10) -> np.ndarray:
    """
    Crop a face region from `img` given a DeepFace-style region dict
    {'x', 'y', 'w', 'h'}.  Adds `pad` pixels of border (clamped to image).
    """
    h_img, w_img = img.shape[:2]
    x  = max(0, region.get("x", 0) - pad)
    y  = max(0, region.get("y", 0) - pad)
    x2 = min(w_img, region.get("x", 0) + region.get("w", 0) + pad)
    y2 = min(h_img, region.get("y", 0) + region.get("h", 0) + pad)
    return img[y:y2, x:x2]
