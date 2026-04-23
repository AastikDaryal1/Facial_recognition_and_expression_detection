"""
main.py
────────
Master orchestrator — runs all pipelines end-to-end in the correct order.

Modes
-----
  train       Run full pipeline: ingest → preprocess → extract → train → deploy
  infer       Run inference on a single image
  realtime    Launch live webcam emotion detector
  api         Start the FastAPI server
  status      Print current model / data status

Usage
-----
    python main.py train
    python main.py train --force-retrain
    python main.py infer --image path/to/photo.jpg
    python main.py realtime
    python main.py api
    python main.py status
"""

from __future__ import annotations

import argparse
import sys
import time

from utils.logger import get_logger

log = get_logger("main")


# ─────────────────────────────────────────────────────────────────────────────
# Sub-commands
# ─────────────────────────────────────────────────────────────────────────────

def cmd_train(args: argparse.Namespace) -> None:
    """Full training pipeline: ingest → preprocess → extract → train → deploy."""
    from pipelines.data_ingestion   import run as ingest
    from pipelines.preprocessing    import run as preprocess
    from pipelines.feature_extraction import run as extract
    from pipelines.training         import run as train
    from pipelines.deployment       import run as deploy

    t0 = time.time()
    log.info("╔══════════════════════════════════════╗")
    log.info("║  FULL TRAINING PIPELINE              ║")
    log.info("╚══════════════════════════════════════╝")

    # ── 1. Data Ingestion ─────────────────────────────────────────────────
    ingest_result = ingest(force_retrain=args.force_retrain)

    if ingest_result["model_loaded"] and not args.force_retrain:
        log.info("✅ Saved model loaded from GCS — nothing to train.")
        log.info("   Use --force-retrain to override.")
        return

    # ── 2. Preprocessing ──────────────────────────────────────────────────
    prep_result = preprocess()
    valid_members = prep_result["valid_members"]

    # ── 3. Feature Extraction ─────────────────────────────────────────────
    fe_result = extract(valid_members=valid_members)

    # ── 4. Training ───────────────────────────────────────────────────────
    train_result = train(
        X            = fe_result["X"],
        y            = fe_result["y"],
        X_raw        = fe_result["X_raw"],
        y_raw        = fe_result["y_raw"],
        valid_members= valid_members,
    )

    # ── 5. Deployment ─────────────────────────────────────────────────────
    if not args.skip_deploy:
        deploy(training_metrics={
            "cv_accuracy"  : train_result["cv_accuracy"],
            "test_accuracy": train_result["test_accuracy"],
        })
    else:
        log.info("Skipping deployment (--skip-deploy).")

    total = time.time() - t0
    log.info("╔══════════════════════════════════════╗")
    log.info("║  ALL PIPELINES COMPLETE  (%.0f s)     ║", total)
    log.info("╚══════════════════════════════════════╝")
    log.info("  CV Accuracy  : %.1f%%", train_result["cv_accuracy"] * 100)
    log.info("  Test Accuracy: %.1f%%", train_result["test_accuracy"] * 100)


def cmd_infer(args: argparse.Namespace) -> None:
    """Run inference on a single image."""
    from pipelines.inference import run as infer

    result = infer(
        image_path = args.image,
        output_dir = args.output_dir,
    )
    log.info("Faces: %d  Identified: %d", result["n_faces"], result["n_identified"])
    for r in result["results"]:
        log.info("  %-20s | %-10s | conf: %.0f%%", r["name"], r["emotion"], r["name_conf"])
    if result["output_path"]:
        log.info("Annotated image saved → %s", result["output_path"])


def cmd_realtime(_args: argparse.Namespace) -> None:
    """Launch the real-time webcam emotion detector."""
    from pipelines.realtime import run as realtime
    realtime()


def cmd_api(_args: argparse.Namespace) -> None:
    """Start the FastAPI server via uvicorn."""
    import uvicorn
    from config.settings import API_HOST, API_PORT, API_WORKERS
    workers = API_WORKERS
    if sys.platform.startswith("win"):
        # Uvicorn multi-worker socket handoff can be flaky on Windows.
        # Default to a single worker for reliable local development.
        workers = 1
    uvicorn.run(
        "api.app:app",
        host    = API_HOST,
        port    = API_PORT,
        workers = workers,
    )


def cmd_status(_args: argparse.Namespace) -> None:
    """Print current data and model status."""
    from config.settings import MODEL_FILES, PROCESSED_DIR, TEAM_FACES_DIR

    log.info("─── Data Status ────────────────────────")
    if TEAM_FACES_DIR.exists():
        members = [d for d in TEAM_FACES_DIR.iterdir() if d.is_dir()]
        log.info("  TeamFaces dir: %s (%d members)", TEAM_FACES_DIR, len(members))
    else:
        log.info("  TeamFaces dir: NOT FOUND — run: python main.py train")

    if PROCESSED_DIR.exists():
        proc = [d for d in PROCESSED_DIR.iterdir() if d.is_dir()]
        log.info("  Processed dir: %s (%d members)", PROCESSED_DIR, len(proc))
    else:
        log.info("  Processed dir: NOT FOUND")

    log.info("─── Model Status ───────────────────────")
    from models.face_model import FaceRecognizer
    if FaceRecognizer.is_available():
        log.info("  Model artefacts: ✅ All present")
        rec = FaceRecognizer()
        log.info("  Members: %s", rec.members)
        log.info("  SVM threshold  : %.2f", rec.svm_threshold)
        log.info("  Cosine threshold: %.2f", rec.cosine_threshold)
    else:
        log.info("  Model artefacts: ❌ Missing")
        for name, path in MODEL_FILES.items():
            mark = "✅" if path.exists() else "❌"
            log.info("    %s  %s", mark, path)


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="main.py",
        description="Face & Emotion Detection — production pipeline orchestrator",
    )
    sub = p.add_subparsers(dest="command", required=True)

    # train
    t = sub.add_parser("train", help="Run the full training pipeline")
    t.add_argument("--force-retrain", action="store_true",
                   help="Ignore saved model and re-train from scratch")
    t.add_argument("--skip-deploy", action="store_true",
                   help="Do not upload model to GCS after training")

    # infer
    i = sub.add_parser("infer", help="Run inference on a single image")
    i.add_argument("--image",      required=True, help="Path to input image")
    i.add_argument("--output-dir", default="data/output", help="Where to save results")

    # realtime
    sub.add_parser("realtime", help="Launch live webcam emotion detector")

    # api
    sub.add_parser("api", help="Start the FastAPI REST server")

    # status
    sub.add_parser("status", help="Print model and data status")

    return p


def main() -> None:
    parser  = build_parser()
    args    = parser.parse_args()
    command = args.command

    dispatch = {
        "train"   : cmd_train,
        "infer"   : cmd_infer,
        "realtime": cmd_realtime,
        "api"     : cmd_api,
        "status"  : cmd_status,
    }
    dispatch[command](args)


if __name__ == "__main__":
    main()
