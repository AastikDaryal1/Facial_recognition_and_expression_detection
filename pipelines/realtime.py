"""
pipelines/realtime.py
──────────────────────
PIPELINE 6 — REAL-TIME WEBCAM EMOTION DETECTOR

Responsibility
--------------
Open the local webcam (cv2.VideoCapture), analyse each frame with
DeepFace, apply the shared infer_emotion() + vote-buffer smoothing,
and display an annotated live feed.

This pipeline is for LOCAL execution (VS Code / desktop).
The original Colab JavaScript approach is replaced with cv2.VideoCapture(0)
which is the standard for non-Colab environments.

Controls
--------
  Press 'q'  →  quit
  Press 's'  →  save current frame to data/output/snapshots/

Usage
-----
    python -m pipelines.realtime
    from pipelines.realtime import run
"""

from __future__ import annotations

import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from deepface import DeepFace

from config.settings import (
    COOLDOWN_SECONDS,
    EMOTION_COLORS_BGR,
    EMOTION_CONFIDENCE_THRESHOLD,
    HARD_SWITCH_THRESHOLD,
    TRACKED_EMOTIONS,
    VOTE_BUFFER_SIZE,
)
from models.emotion_model import infer_emotion
from utils.logger import get_logger

log = get_logger(__name__)

_SNAPSHOT_DIR = Path("data/output/snapshots")
_SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Frame annotation
# ─────────────────────────────────────────────────────────────────────────────

def _annotate_frame(
    frame           : np.ndarray,
    displayed_emotion: str,
    inferred_emotion : str,
    inferred_conf    : float,
    dominant_raw     : str,
    effective_conf   : float,
    emotion_scores   : dict,
    region           : dict,
    frame_count      : int,
) -> np.ndarray:
    """Overlay all debug info on the frame in-place."""
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Face bounding box
    if region:
        rx, ry = region.get("x", 0), region.get("y", 0)
        rw, rh = region.get("w", 0), region.get("h", 0)
        if rw > 0 and rh > 0:
            cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 2)

    # Main emotion label
    label_color = EMOTION_COLORS_BGR.get(displayed_emotion, (0, 255, 0))
    cv2.putText(frame, f"Emotion: {displayed_emotion}",
                (20, 45), font, 1.0, label_color, 2, cv2.LINE_AA)

    # Per-emotion score bars
    filtered = {k: emotion_scores[k] for k in TRACKED_EMOTIONS if k in emotion_scores}
    for i, (emo, score) in enumerate(filtered.items()):
        cv2.putText(frame, f"{emo}: {score:.1f}%",
                    (20, 90 + i * 28), font, 0.58, (0, 255, 255), 2, cv2.LINE_AA)

    # Debug line
    conf_color = (0, 255, 0) if effective_conf >= EMOTION_CONFIDENCE_THRESHOLD else (0, 100, 255)
    cv2.putText(
        frame,
        f"Inferred: {inferred_emotion} ({inferred_conf:.1f}%)  Raw: {dominant_raw}",
        (20, 90 + len(filtered) * 28 + 8), font, 0.48, conf_color, 2, cv2.LINE_AA,
    )

    # Frame counter + controls
    h = frame.shape[0]
    cv2.putText(frame, f"Frame: {frame_count}  |  q=quit  s=snapshot",
                (10, h - 10), font, 0.42, (180, 180, 180), 1, cv2.LINE_AA)
    return frame


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────

def run(camera_index: int = 0, window_name: str = "Emotion Detector") -> None:
    """
    Launch the real-time webcam emotion detector.

    Parameters
    ----------
    camera_index : int   OpenCV camera index (0 = default webcam)
    window_name  : str   OpenCV window title
    """
    log.info("=" * 60)
    log.info("PIPELINE 6 — REAL-TIME WEBCAM")
    log.info("=" * 60)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open camera at index {camera_index}. "
            "Check that a webcam is connected."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    log.info("Camera opened — press 'q' to quit, 's' to save snapshot.")

    # ── State ──────────────────────────────────────────────────────────────
    emotion_buffer    : list[str] = []
    displayed_emotion : str       = "Detecting..."
    last_switch_time  : float     = time.time()
    frame_count       : int       = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                log.warning("Failed to grab frame — retrying …")
                time.sleep(0.05)
                continue

            frame_count += 1

            # ── DeepFace analysis ─────────────────────────────────────────
            region         = {}
            emotion_scores = {}
            dominant_raw   = "unknown"
            inferred_emotion, inferred_conf = displayed_emotion, 0.0

            try:
                result = DeepFace.analyze(
                    frame,
                    actions=["emotion"],
                    enforce_detection=False,
                )
                if isinstance(result, dict):
                    result = [result]

                if result:
                    first          = result[0]
                    emotion_scores = first["emotion"]
                    dominant_raw   = first["dominant_emotion"]
                    region         = first.get("region", {})

                    inferred_emotion, inferred_conf = infer_emotion(emotion_scores)
                    effective_conf = inferred_conf

                    # ── Vote buffer ───────────────────────────────────────
                    if effective_conf >= EMOTION_CONFIDENCE_THRESHOLD:
                        # Hard-switch: clear buffer if very confident new emotion
                        if effective_conf >= HARD_SWITCH_THRESHOLD and emotion_buffer:
                            majority = Counter(emotion_buffer).most_common(1)[0][0]
                            if inferred_emotion != majority:
                                emotion_buffer.clear()
                        emotion_buffer.append(inferred_emotion)

                    if len(emotion_buffer) > VOTE_BUFFER_SIZE:
                        emotion_buffer.pop(0)

                    if emotion_buffer:
                        voted = Counter(emotion_buffer).most_common(1)[0][0]
                        now   = time.time()
                        if voted != displayed_emotion and (now - last_switch_time) >= COOLDOWN_SECONDS:
                            log.info("Emotion switched: %s → %s (conf=%.1f%%)",
                                     displayed_emotion, voted, inferred_conf)
                            displayed_emotion = voted
                            last_switch_time  = now
                else:
                    emotion_buffer.clear()
                    cv2.putText(frame, "No face detected",
                                (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            except Exception as exc:
                log.debug("DeepFace error: %s", exc)

            # ── Annotate + display ────────────────────────────────────────
            frame = _annotate_frame(
                frame, displayed_emotion, inferred_emotion,
                inferred_conf, dominant_raw, inferred_conf,
                emotion_scores, region, frame_count,
            )

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                log.info("Quit key pressed.")
                break
            elif key == ord("s"):
                snap_path = _SNAPSHOT_DIR / f"snap_{frame_count:06d}.jpg"
                cv2.imwrite(str(snap_path), frame)
                log.info("Snapshot saved → %s", snap_path)

    except KeyboardInterrupt:
        log.info("Interrupted by user (Ctrl+C).")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        log.info("Camera released.  Frames processed: %d", frame_count)


if __name__ == "__main__":
    run()
