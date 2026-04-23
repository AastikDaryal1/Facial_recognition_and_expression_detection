"""
models/face_model.py
─────────────────────
Wrapper around the trained SVM + cosine gate for face identity recognition.

FaceRecognizer loads the persisted artefacts (team_face_model.pkl,
team_label_encoder.pkl, team_mean_embeddings.pkl, team_config.json)
and exposes a single `.predict(embedding)` method that applies both
the SVM probability gate AND the cosine-similarity gate.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from config.settings import (
    COSINE_SIMILARITY_THRESHOLD,
    MODEL_FILES,
    SVM_CONFIDENCE_THRESHOLD,
)
from utils.logger import get_logger

log = get_logger(__name__)


class FaceRecognizer:
    """
    Loads trained model artefacts and identifies a face from its
    FaceNet embedding.

    Dual-gate strategy
    ------------------
    A face is labelled as a known member ONLY when:
      svm_probability  >= SVM_CONFIDENCE_THRESHOLD  AND
      cosine_similarity >= COSINE_SIMILARITY_THRESHOLD

    If either gate fails → "UNKNOWN".
    """

    def __init__(
        self,
        model_path      : Path = MODEL_FILES["face_model"],
        encoder_path    : Path = MODEL_FILES["label_encoder"],
        mean_embs_path  : Path = MODEL_FILES["mean_embeddings"],
        config_path     : Path = MODEL_FILES["config"],
        svm_threshold   : float = SVM_CONFIDENCE_THRESHOLD,
        cosine_threshold: float = COSINE_SIMILARITY_THRESHOLD,
    ) -> None:
        self.svm_threshold    = svm_threshold
        self.cosine_threshold = cosine_threshold

        log.info("Loading FaceRecognizer artefacts …")
        with open(model_path,     "rb") as f: self.model  = pickle.load(f)
        with open(encoder_path,   "rb") as f: self.le     = pickle.load(f)
        with open(mean_embs_path, "rb") as f: mean_embs   = pickle.load(f)
        with open(config_path,    "r")  as f: self.config = json.load(f)

        # Build mean-embedding matrix ordered by le.classes_
        self.members      = self.le.classes_.tolist()
        self.mean_matrix  = np.array(
            [mean_embs[m] for m in self.members], dtype=np.float32
        )
        log.info("FaceRecognizer ready — members: %s", self.members)

    # ── Public API ────────────────────────────────────────────────────────

    def predict(self, embedding: np.ndarray) -> dict:
        """
        Identify a face from its L2-normalised FaceNet embedding.

        Parameters
        ----------
        embedding : np.ndarray  shape (512,) or (1, 512)

        Returns
        -------
        dict with keys:
            name        – str    : member name or "UNKNOWN"
            confidence  – float  : SVM probability * 100
            svm_prob    – float  : raw SVM probability [0,1]
            cosine_sim  – float  : cosine similarity [0,1]
            candidate   – str    : best SVM candidate (even if rejected)
        """
        emb = normalize(embedding.reshape(1, -1))[0]

        prob     = self.model.predict_proba([emb])[0]
        best_idx = int(np.argmax(prob))
        svm_prob = float(prob[best_idx])
        candidate = self.members[best_idx]

        cos_sims = cosine_similarity([emb], self.mean_matrix)[0]
        best_cos = float(cos_sims[best_idx])

        log.debug(
            "Identity — candidate=%s  svm=%.3f  cos=%.3f",
            candidate, svm_prob, best_cos,
        )

        if svm_prob >= self.svm_threshold and best_cos >= self.cosine_threshold:
            name = candidate
        else:
            name = "UNKNOWN"

        return {
            "name"      : name,
            "confidence": svm_prob * 100,
            "svm_prob"  : svm_prob,
            "cosine_sim": best_cos,
            "candidate" : candidate,
        }

    @classmethod
    def is_available(cls) -> bool:
        """Return True only if all required model files exist on disk."""
        return all(p.exists() for p in [
            MODEL_FILES["face_model"],
            MODEL_FILES["label_encoder"],
            MODEL_FILES["mean_embeddings"],
            MODEL_FILES["config"],
        ])
