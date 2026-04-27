"""
tests/test_pipelines.py
────────────────────────
Unit tests for the core models and pipeline helpers.

Run with:
    pytest tests/ -v
    pytest tests/ -v --tb=short -q   (quiet mode)
"""

from __future__ import annotations

import numpy as np
import pytest

# ─────────────────────────────────────────────────────────────────────────────
# models/emotion_model.py
# ─────────────────────────────────────────────────────────────────────────────

from models.emotion_model import group_emotion, infer_emotion


class TestGroupEmotion:
    def test_disgust_maps_to_angry(self):
        assert group_emotion("disgust") == "angry"

    def test_fear_not_in_groups_returns_itself(self):
        # 'fear' is handled by rules in infer_emotion, not by groups
        # group_emotion falls through to return original
        result = group_emotion("fear")
        assert result == "fear"

    def test_happy_maps_to_happy(self):
        assert group_emotion("happy") == "happy"

    def test_unknown_emotion_returns_itself(self):
        assert group_emotion("unknown_xyz") == "unknown_xyz"


class TestInferEmotion:
    """Tests for the 4-rule emotion inference logic."""

    def _scores(self, **kwargs) -> dict:
        base = {e: 0.0 for e in ["happy", "neutral", "surprise", "angry", "disgust", "sad", "fear"]}
        base.update(kwargs)
        return base

    # ── Rule 0: SURPRISE ──────────────────────────────────────────────────
    def test_rule0_surprise_signal_above_threshold(self):
        scores = self._scores(surprise=30.0)
        label, conf = infer_emotion(scores)
        assert label == "surprise"
        assert conf >= 25

    def test_rule0_fear_boosts_surprise(self):
        # fear * 0.6 = 18 → surprise_signal = 8 + 18 = 26 ≥ 25
        scores = self._scores(surprise=8.0, fear=30.0)
        label, _ = infer_emotion(scores)
        assert label == "surprise"

    # ── Rule 1: SAD ───────────────────────────────────────────────────────
    def test_rule1_sad_detected(self):
        scores = self._scores(sad=15.0, neutral=50.0)
        label, _ = infer_emotion(scores)
        assert label == "sad"

    def test_rule1_sad_blocked_by_high_happy(self):
        scores = self._scores(sad=15.0, happy=20.0)
        label, _ = infer_emotion(scores)
        assert label != "sad"

    # ── Rule 2: ANGRY ─────────────────────────────────────────────────────
    def test_rule2_angry_detected(self):
        scores = self._scores(angry=25.0)
        label, _ = infer_emotion(scores)
        assert label == "angry"

    def test_rule2_angry_via_disgust(self):
        # angry_signal = 0 + 50*0.5 = 25 ≥ 20
        scores = self._scores(disgust=50.0)
        label, _ = infer_emotion(scores)
        assert label == "angry"

    # ── Rule 3: NEUTRAL > HAPPY suppression ──────────────────────────────
    def test_rule3_weak_happy_suppressed_to_neutral(self):
        scores = self._scores(happy=35.0, neutral=30.0)
        label, _ = infer_emotion(scores)
        assert label == "neutral"

    def test_rule3_strong_happy_not_suppressed(self):
        scores = self._scores(happy=60.0, neutral=25.0)
        label, _ = infer_emotion(scores)
        assert label == "happy"

    # ── Default ───────────────────────────────────────────────────────────
    def test_default_returns_dominant(self):
        scores = self._scores(happy=80.0)
        label, conf = infer_emotion(scores)
        assert label == "happy"
        assert conf == 80.0


# ─────────────────────────────────────────────────────────────────────────────
# utils/image_utils.py
# ─────────────────────────────────────────────────────────────────────────────

from utils.image_utils import crop_face, resize_to_width


class TestImageUtils:
    def test_resize_to_width_downscales(self):
        import numpy as np
        img = np.zeros((480, 1280, 3), dtype=np.uint8)
        result = resize_to_width(img, max_width=640)
        assert result.shape[1] == 640
        assert result.shape[0] == 240   # aspect ratio preserved

    def test_resize_to_width_no_op_when_smaller(self):
        import numpy as np
        img = np.zeros((240, 320, 3), dtype=np.uint8)
        result = resize_to_width(img, max_width=640)
        assert result.shape == img.shape

    def test_crop_face_clamps_to_image_bounds(self):
        import numpy as np
        img    = np.zeros((100, 100, 3), dtype=np.uint8)
        region = {"x": 0, "y": 0, "w": 50, "h": 50}
        crop   = crop_face(img, region, pad=20)
        assert crop.shape[0] > 0 and crop.shape[1] > 0
        # Should not raise even with large pad


# ─────────────────────────────────────────────────────────────────────────────
# config/settings.py — smoke test
# ─────────────────────────────────────────────────────────────────────────────

class TestSettings:
    def test_emotion_groups_covers_all_tracked(self):
        from config.settings import EMOTION_GROUPS, TRACKED_EMOTIONS
        grouped = {e for members in EMOTION_GROUPS.values() for e in members}
        # Every group member should be a tracked emotion
        assert grouped.issubset(set(TRACKED_EMOTIONS))

    def test_thresholds_are_in_valid_range(self):
        from config.settings import (
            COSINE_SIMILARITY_THRESHOLD,
            EMOTION_CONFIDENCE_THRESHOLD,
            SVM_CONFIDENCE_THRESHOLD,
        )
        assert 0.0 < SVM_CONFIDENCE_THRESHOLD    <= 1.0
        assert 0.0 < COSINE_SIMILARITY_THRESHOLD <= 1.0
        assert 0.0 < EMOTION_CONFIDENCE_THRESHOLD <= 100.0

    def test_model_files_dict_has_required_keys(self):
        from config.settings import MODEL_FILES
        for key in ["face_model", "label_encoder", "mean_embeddings", "config", "embed_cache"]:
            assert key in MODEL_FILES
