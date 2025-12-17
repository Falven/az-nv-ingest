from __future__ import annotations

import logging
from typing import Optional


def configure_logging(level: str = "INFO") -> logging.Logger:
    """
    Configure root logging once with a simple, consistent formatter.

    Args:
        level: Logging level string understood by ``logging`` (e.g., ``INFO``).

    Returns:
        The configured root logger.
    """
    logger = logging.getLogger()
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level.upper())
    return logger


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Retrieve a module logger with an optional explicit level.

    Args:
        name: Logger name, typically ``__name__``.
        level: Optional logging level string.

    Returns:
        The configured logger instance.
    """
    logger = logging.getLogger(name)
    if level is not None:
        logger.setLevel(level.upper())
    return logger
