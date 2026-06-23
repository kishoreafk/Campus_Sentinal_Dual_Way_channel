"""Loguru-based structured logger."""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger as _logger

from config.settings import get_settings

_CONFIGURED = False


def setup_logger(name: str = "cctv") -> None:
    """Configure the global logger. Safe to call multiple times."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    settings = get_settings()
    _logger.remove()
    _logger.add(
        sys.stderr,
        level=settings.log_level,
        colorize=True,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
    )
    # File log (rotated daily)
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    _logger.add(
        log_dir / "cctv_{time:YYYY-MM-DD}.log",
        level=settings.log_level,
        rotation="00:00",
        retention="30 days",
        compression="zip",
        format=(
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
            "{name}:{function}:{line} - {message}"
        ),
    )
    _CONFIGURED = True


def get_logger():
    """Return the configured logger."""
    if not _CONFIGURED:
        setup_logger()
    return _logger


# Module-level logger for convenience
logger = get_logger()
