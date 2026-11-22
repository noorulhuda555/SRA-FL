"""Logging utilities for SRA-FL."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logging(log_dir: str | Path | None = None, level: int = logging.INFO) -> None:
    """Configure root logging once for console and optional file output."""
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if log_dir is not None:
        path = Path(log_dir)
        path.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(path / "sra_fl.log", encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    """Return a named logger."""
    return logging.getLogger(name)
