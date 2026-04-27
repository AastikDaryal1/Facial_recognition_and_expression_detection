"""
models/emotion_model.py
────────────────────────
Core emotion-inference logic, extracted from the original notebook.

Provides
--------
group_emotion(emotion)              → grouped label string
infer_emotion(emotion_scores)       → (label, confidence)
detect_with_calibration(img_path)   → (label, scores_dict)

These are pure functions — no state, no side effects.
They are imported by both the inference pipeline and the real-time pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from config.settings import EMOTION_GROUPS
from utils.logger import get_logger

log = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Emotion grouping
# ─────────────────────────────────────────────────────────────────────────────

def group_emotion(emotion: str) -> str:
    """
    Map a raw DeepFace label to a merged group label.
    e.g. 'disgust' → 'angry',  'fear' → stays as 'fear' (handled by rules)
    """
    for group_label, members in EMOTION_GROUPS.items():
        if emotion in members:
            return group_label
    return emotion


# ─────────────────────────────────────────────────────────────────────────────
# Smart emotion inference
# ─────────────────────────────────────────────────────────────────────────────

def infer_emotion(emotion_scores: dict[str, float]) -> tuple[str, float]:
    """
    Corrects DeepFace's known weaknesses using four hand-crafted rules.

    Rule 0 — SURPRISE  : runs first; claims 'fear' votes before sad-rule.
    Rule 1 — SAD       : catches sad+fear combined signal on low-energy faces.
    Rule 2 — ANGRY     : catches angry+disgust combined signal.
    Rule 3 — NEUTRAL/HAPPY: suppresses weak happy on resting faces.
    Default — trust dominant emotion for everything else.

    Parameters
    ----------
    emotion_scores : dict  e.g. {'happy': 72.3, 'neutral': 18.1, …}

    Returns
    -------
    (emotion_label, confidence_score)
    """
    happy    = emotion_scores.get("happy",    0.0)
    neutral  = emotion_scores.get("neutral",  0.0)
    surprise = emotion_scores.get("surprise", 0.0)
    angry    = emotion_scores.get("angry",    0.0)
    disgust  = emotion_scores.get("disgust",  0.0)
    sad      = emotion_scores.get("sad",      0.0)
    fear     = emotion_scores.get("fear",     0.0)

    # Rule 0: SURPRISE
    surprise_signal = surprise + (fear * 0.6)
    if surprise_signal >= 25:
        return "surprise", surprise_signal

    # Rule 1: SAD
    sad_signal = sad + (fear * 0.4)
    if sad_signal >= 12 and happy < 10 and angry < 25 and surprise < 20 and neutral < 85:
        return "sad", sad_signal

    # Rule 2: ANGRY
    angry_signal = angry + (disgust * 0.5)
    if angry_signal >= 20 and happy < 15 and surprise < 15 and sad < 15:
        return "angry", angry_signal

    # Rule 3: NEUTRAL vs HAPPY suppression
    if happy > neutral and happy < 45 and neutral > 20:
        return "neutral", neutral

    # Default
    dominant = max(emotion_scores, key=emotion_scores.get)
    return group_emotion(dominant), emotion_scores[dominant]


# ─────────────────────────────────────────────────────────────────────────────
# Calibrated detection (wraps DeepFace)
# ─────────────────────────────────────────────────────────────────────────────

def detect_with_calibration(
    img_path: str | Path | np.ndarray,
    detector_backend: str = "skip",
) -> tuple[Optional[str], dict]:
    """
    Run DeepFace emotion analysis and return calibrated results.

    Parameters
    ----------
    img_path         : path to the image file, or a cropped BGR numpy array
    detector_backend : DeepFace detector backend (use 'skip' for cropped faces)

    Returns
    -------
    (dominant_emotion: str | None, scores: dict)
    """
    try:
        from deepface import DeepFace  # lazy import — not every module needs it
        
        # Pass numpy arrays directly, stringify paths
        target_img = img_path if isinstance(img_path, np.ndarray) else str(img_path)

        result = DeepFace.analyze(
            img_path          = target_img,
            actions           = ["emotion"],
            detector_backend  = detector_backend,
            enforce_detection = False,
            silent            = True,
        )
        if isinstance(result, dict):
            result = [result]
        if not result:
            return None, {}

        scores         = result[0]["emotion"]
        dominant, conf = infer_emotion(scores)
        return dominant, scores

    except Exception as exc:
        log.warning("detect_with_calibration error: %s", exc)
        return None, {}
