"""Timestamped console output and file logging utilities."""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from typing import Optional, TextIO


def tprint(*args: object, **kwargs: object) -> None:
    """
    Print with a ``[HH:MM:SS]`` timestamp prefix.

    Args:
        *args: Pos arguments to ``print``.
        **kwargs: Kwd arguments to ``print``
    """
    ts: str = datetime.now().strftime("[%H:%M:%S]")
    print(ts, *args, **kwargs)


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Create or retrieve a logger with console and optional file output.

    Args:
        name: Logger name (typically ``__name__``).
        log_file: If provided, also write log records to this file path.
        level: Logging level (default ``logging.INFO``).

    Returns:
        A configured class `logging.Logger` instance.

    Example:
        >>> logger = setup_logger(__name__, "train.log")
        >>> logger.info("training started")
    """
    logger: logging.Logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        fmt: logging.Formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s - %(message)s",
            datefmt="%H:%M:%S",
        )
        console: logging.StreamHandler[TextIO] = logging.StreamHandler(sys.stdout)
        console.setFormatter(fmt)
        logger.addHandler(console)

        if log_file is not None:
            file_handler: logging.FileHandler = logging.FileHandler(log_file)
            file_handler.setFormatter(fmt)
            logger.addHandler(file_handler)

    return logger
