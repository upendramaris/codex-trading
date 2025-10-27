"""
Logging helper that standardises log format across the project and supports file output.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
_LOG_CONFIGURED = False


def configure_logging(log_file: Optional[str] = None, level: int = logging.INFO) -> None:
    """
    Configure root logging with optional rotating file handler.

    Subsequent calls update the root level and add missing file handlers without duplicating streams.
    """

    global _LOG_CONFIGURED

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    if not _LOG_CONFIGURED:
        formatter = logging.Formatter(_LOG_FORMAT)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        root_logger.addHandler(stream_handler)
        _LOG_CONFIGURED = True

    if log_file:
        file_path = Path(log_file)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        existing = [
            handler
            for handler in logging.getLogger().handlers
            if isinstance(handler, logging.FileHandler) and getattr(handler, "baseFilename", None) == str(file_path.resolve())
        ]
        if not existing:
            file_handler = RotatingFileHandler(file_path, maxBytes=10_000_000, backupCount=5)
            file_handler.setFormatter(logging.Formatter(_LOG_FORMAT))
            file_handler.setLevel(level)
            root_logger.addHandler(file_handler)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a configured logger instance."""
    if not _LOG_CONFIGURED:
        configure_logging()
    return logging.getLogger(name)
