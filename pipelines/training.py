"""
pipelines/training.py
──────────────────────
PIPELINE 4 — TRAINING

Responsibility
--------------
Train a Calibrated SVM on the FaceNet embeddings produced by Pipeline 3,
evaluate it, compute per-member mean embeddings (used as a cosine gate),
and persist all model artefacts locally.

Steps
-----
1. Encode labels with LabelEncoder.
2. Stratified train/test split.
3. Fit CalibratedClassifierCV(SVC(rbf)) on training set.
4. Evaluate: accuracy, CV score, classification report, confusion matrix.
5. Compute per-member mean embedding vectors.
6. Persist: team_face_model.pkl, team_label_encoder.pkl,
            team_mean_embeddings.pkl, team_config.json.

Usage
-----
    python -m pipelines.training
    from pipelines.training import run
"""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

from config.settings import (
    COSINE_SIMILARITY_THRESHOLD,
    EMOTION_BOX_COLORS_RGB,
    MODEL_FILES,
    RANDOM_STATE,
    SVM_C,
    SVM_CONFIDENCE_THRESHOLD,
    SVM_KERNEL,
    TRAIN_TEST_SPLIT,
)
from utils.logger import get_logger

log = get_logger(__name__)


def run(
    X     : np.ndarray,
    y     : np.ndarray,
    X_raw : np.ndarray,
    y_raw : np.ndarray,
    valid_members: Optional[list[str]] = None,
) -> dict:
    """
    Execute the training pipeline.

    Parameters
    ----------
    X, y        : augmented embeddings + labels (from feature_extraction)
    X_raw, y_raw: original (pre-augmentation) embeddings + labels
    valid_members: ordered list of member names (used for mean-embeddings)

    Returns
    -------
    dict with keys:
        team_model      – fitted CalibratedClassifierCV
        le_team         – fitted LabelEncoder
        mean_embeddings – dict {member: np.ndarray}
        cv_accuracy     – float
        test_accuracy   – float
        elapsed_s       – float
    """
    t0 = time.time()
    log.info("=" * 60)
    log.info("PIPELINE 4 — TRAINING")
    log.info("=" * 60)

    # ── Step 1: Encode labels ─────────────────────────────────────────────
    log.info("Step 1: Encoding labels …")
    le_team = LabelEncoder()
    y_enc   = le_team.fit_transform(y)
    log.info("  Classes: %s", le_team.classes_.tolist())

    # ── Step 2: Stratified split ──────────────────────────────────────────
    log.info("Step 2: Stratified train/test split (test=%.0f%%) …", TRAIN_TEST_SPLIT * 100)
    stratify = y_enc if len(set(y_enc)) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc,
        test_size=TRAIN_TEST_SPLIT,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )
    log.info("  Train=%d  Test=%d", len(X_train), len(X_test))

    # ── Step 3: Fit SVM ───────────────────────────────────────────────────
    log.info("Step 3: Training CalibratedSVC (kernel=%s, C=%.1f) …", SVM_KERNEL, SVM_C)
    svm_base   = SVC(kernel=SVM_KERNEL, C=SVM_C, probability=False, random_state=RANDOM_STATE)
    team_model = CalibratedClassifierCV(svm_base, method="isotonic", cv=3)
    team_model.fit(X_train, y_train)
    log.info("  Training complete.")

    # ── Step 4: Evaluate ──────────────────────────────────────────────────
    log.info("Step 4: Evaluating …")
    cv_scores   = cross_val_score(team_model, X, y_enc, cv=3, scoring="accuracy")
    cv_accuracy = float(cv_scores.mean())
    log.info("  CV Accuracy : %.1f%% ± %.1f%%", cv_accuracy * 100, cv_scores.std() * 100)

    y_pred        = team_model.predict(X_test)
    test_accuracy = float(accuracy_score(y_test, y_pred))
    log.info("  Test Accuracy: %.1f%%", test_accuracy * 100)

    report = classification_report(y_test, y_pred, target_names=le_team.classes_, output_dict=True)
    log.info("Classification report:\n%s",
             classification_report(y_test, y_pred, target_names=le_team.classes_))

    cm = confusion_matrix(y_test, y_pred)
    log.info("Confusion matrix:\n%s", cm)

    # ── Step 5: Mean embeddings for cosine gate ───────────────────────────
    log.info("Step 5: Computing per-member mean embeddings …")
    if valid_members is None:
        valid_members = le_team.classes_.tolist()
    mean_embeddings: dict[str, list] = {}
    for member in valid_members:
        mask = y_raw == member
        if mask.sum() == 0:
            log.warning("  No raw embeddings for %s — skipping mean embedding", member)
            continue
        mean_embeddings[member] = X_raw[mask].mean(axis=0).tolist()
    log.info("  Mean embeddings computed for %d members.", len(mean_embeddings))

    # ── Step 6: Persist artefacts ─────────────────────────────────────────
    log.info("Step 6: Saving model artefacts …")
    MODEL_FILES["face_model"].parent.mkdir(parents=True, exist_ok=True)

    with open(MODEL_FILES["face_model"],      "wb") as f: pickle.dump(team_model,      f)
    with open(MODEL_FILES["label_encoder"],   "wb") as f: pickle.dump(le_team,          f)
    with open(MODEL_FILES["mean_embeddings"], "wb") as f: pickle.dump(mean_embeddings,  f)

    config_payload = {
        "members"           : le_team.classes_.tolist(),
        "threshold"         : SVM_CONFIDENCE_THRESHOLD,
        "cosine_threshold"  : COSINE_SIMILARITY_THRESHOLD,
        "n_embeddings"      : len(X_raw),
        "cv_accuracy"       : cv_accuracy,
        "test_accuracy"     : test_accuracy,
        "emotion_box_colors": {k: list(v) for k, v in EMOTION_BOX_COLORS_RGB.items()},
    }
    with open(MODEL_FILES["config"], "w") as f:
        json.dump(config_payload, f, indent=2)

    log.info("  ✅ Saved: %s", [str(v) for v in MODEL_FILES.values()])
    elapsed = time.time() - t0
    log.info("Training complete in %.1f s", elapsed)

    return {
        "team_model"      : team_model,
        "le_team"         : le_team,
        "mean_embeddings" : mean_embeddings,
        "cv_accuracy"     : cv_accuracy,
        "test_accuracy"   : test_accuracy,
        "report"          : report,
        "elapsed_s"       : elapsed,
    }


if __name__ == "__main__":
    # Standalone: re-use feature extraction output
    from pipelines.feature_extraction import run as fe_run
    fe = fe_run()
    run(fe["X"], fe["y"], fe["X_raw"], fe["y_raw"])
