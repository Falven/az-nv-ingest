from __future__ import annotations

from oim_common.errors import InferenceError, InvalidImageError

__all__ = [
    "InvalidImageError",
    "InferenceError",
    "TritonInferenceError",
    "TritonStartupError",
]


class TritonInferenceError(RuntimeError):
    """Raised when Triton inference fails."""


class TritonStartupError(RuntimeError):
    """Raised when the embedded Triton server fails to start."""
