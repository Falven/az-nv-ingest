from __future__ import annotations

from oim_common.errors import InferenceError, InvalidImageError

__all__ = [
    "InvalidImageError",
    "InferenceError",
    "TritonStartupError",
    "TritonInferenceError",
]


class TritonStartupError(Exception):
    """Raised when the embedded Triton server cannot start or become ready."""


class TritonInferenceError(Exception):
    """Raised when Triton inference fails or returns malformed data."""
