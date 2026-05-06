"""
utils/validation.py
───────────────────
Production-grade validation logic for face detection and quality gates.
"""

from __future__ import annotations
import cv2
import numpy as np
from typing import Optional
from config.settings import (
    FACE_CONFIDENCE_THRESHOLD,
    MIN_FACE_WIDTH,
    MIN_FACE_HEIGHT,
    MAX_FACE_AREA_FRACTION,
    FACE_BLUR_THRESHOLD,
)
from utils.logger import get_logger

log = get_logger(__name__)

def validate_face_region(
    region: dict, 
    img_bgr: np.ndarray, 
    img_shape: tuple[int, int],
    face_idx: int = 0
) -> tuple[bool, Optional[str]]:
    """
    Validates a face detection region based on production thresholds.
    Checks: Confidence, Dimensions, Frame Area, Aspect Ratio, and Blur.
    """
    x, y, w, h = region.get("x", 0), region.get("y", 0), region.get("w", 0), region.get("h", 0)
    conf = region.get("confidence", 1.0)
    h_img, w_img = img_shape
    
    # 1. Detector Confidence
    if conf < FACE_CONFIDENCE_THRESHOLD:
        return False, f"Low detection confidence ({conf:.2f} < {FACE_CONFIDENCE_THRESHOLD})"
    
    # 2. Absolute Dimensions (80x80 px)
    if w < MIN_FACE_WIDTH or h < MIN_FACE_HEIGHT:
        return False, f"Face too small ({w}x{h} < {MIN_FACE_WIDTH}x{MIN_FACE_HEIGHT})"
    
    # 3. Frame Area Coverage (Max 45%)
    face_area = w * h
    img_area = w_img * h_img
    area_fraction = face_area / img_area
    if area_fraction > MAX_FACE_AREA_FRACTION:
        return False, f"Face area too large ({area_fraction:.2%} > {MAX_FACE_AREA_FRACTION:.0%})"
    if area_fraction < 0.005: # Hard floor of 0.5% area
        return False, f"Face area too small ({area_fraction:.2%})"

    # 4. Aspect Ratio (Roughly square)
    aspect_ratio = w / h
    if aspect_ratio < 0.6 or aspect_ratio > 1.6:
        return False, f"Invalid aspect ratio ({aspect_ratio:.2f})"
    
    # 5. Blur Detection (Laplacian Variance)
    try:
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(w_img, x + w), min(h_img, y + h)
        face_crop = img_bgr[y1:y2, x1:x2]
        if face_crop.size == 0:
            return False, "Empty crop"
            
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        if blur_score < FACE_BLUR_THRESHOLD:
            return False, f"Face too blurry ({blur_score:.1f} < {FACE_BLUR_THRESHOLD})"
    except Exception as e:
        return False, f"Blur validation error: {str(e)}"

    return True, None
