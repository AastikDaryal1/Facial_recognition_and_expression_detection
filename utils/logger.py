"""
utils/logger.py
───────────────
Centralised, structured logger used by every pipeline.

Features
--------
- Console handler (coloured, human-readable)
- Rotating file handler  →  logs/app.log
- JSON file handler      →  logs/app.jsonl  (one JSON object per line)
  → Easy to ingest into ELK / Cloud Logging / Datadog

Usage
-----
    from utils.logger import get_logger
    log = get_logger(__name__)
    log.info("Training started", extra={"epoch": 1, "lr": 0.001})
"""

import json
import logging
import logging.handlers
import sys
from datetime import datetime, timezone
from pathlib import Path

from config.settings import LOG_LEVEL, LOGS_DIR


# ─────────────────────────────────────────────────────────────────────────────
# ANSI colour codes for console output
# ─────────────────────────────────────────────────────────────────────────────
_COLOURS = {
    "DEBUG"   : "\033[36m",   # cyan
    "INFO"    : "\033[32m",   # green
    "WARNING" : "\033[33m",   # yellow
    "ERROR"   : "\033[31m",   # red
    "CRITICAL": "\033[35m",   # magenta
    "RESET"   : "\033[0m",
}


class _ColouredFormatter(logging.Formatter):
    """Human-readable coloured formatter for the console."""

    FMT = "%(asctime)s  %(levelname)-8s  %(name)s  |  %(message)s"

    def format(self, record: logging.LogRecord) -> str:
        colour = _COLOURS.get(record.levelname, "")
        reset  = _COLOURS["RESET"]
        record.levelname = f"{colour}{record.levelname}{reset}"
        return super().format(record)


class _JsonFormatter(logging.Formatter):
    """
    One-JSON-object-per-line formatter for machine consumption.
    Any key/value pairs passed as `extra={}` are included at the
    top level of the JSON object.
    """

    _SKIP = {
        "name", "msg", "args", "levelname", "levelno", "pathname",
        "filename", "module", "exc_info", "exc_text", "stack_info",
        "lineno", "funcName", "created", "msecs", "relativeCreated",
        "thread", "threadName", "processName", "process", "message",
        "taskName",
    }

    def format(self, record: logging.LogRecord) -> str:
        record.message = record.getMessage()
        payload = {
            "ts"      : datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level"   : record.levelname,
            "logger"  : record.name,
            "message" : record.message,
            "module"  : record.module,
            "line"    : record.lineno,
        }
        # Include any extra fields the caller passed
        for k, v in record.__dict__.items():
            if k not in self._SKIP:
                payload[k] = v
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


# ─────────────────────────────────────────────────────────────────────────────
# Public factory
# ─────────────────────────────────────────────────────────────────────────────
_ROOT_LOGGER_CONFIGURED = False


def _configure_root() -> None:
    global _ROOT_LOGGER_CONFIGURED
    if _ROOT_LOGGER_CONFIGURED:
        return

    root = logging.getLogger()
    root.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))

    # 1. Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(
        _ColouredFormatter(
            fmt="%(asctime)s  %(levelname)-8s  %(name)s  |  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root.addHandler(ch)

    # 2. Rotating plain-text file handler  →  logs/app.log
    fh = logging.handlers.RotatingFileHandler(
        LOGS_DIR / "app.log",
        maxBytes=10 * 1024 * 1024,   # 10 MB
        backupCount=5,
        encoding="utf-8",
    )
    fh.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s  %(levelname)-8s  %(name)s  |  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root.addHandler(fh)

    # 3. JSON file handler  →  logs/app.jsonl
    jh = logging.handlers.RotatingFileHandler(
        LOGS_DIR / "app.jsonl",
        maxBytes=20 * 1024 * 1024,   # 20 MB
        backupCount=5,
        encoding="utf-8",
    )
    jh.setFormatter(_JsonFormatter())
    root.addHandler(jh)

    _ROOT_LOGGER_CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a named logger, configuring the root logger on first call."""
    _configure_root()
    return logging.getLogger(name)
