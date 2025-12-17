from __future__ import annotations


class InvalidImageError(ValueError):
    """Raised when an input image cannot be decoded."""


class InferenceError(RuntimeError):
    """Raised for unexpected failures during inference preparation."""


class TritonInferenceError(RuntimeError):
    """Raised when Triton inference fails."""


class TritonStartupError(RuntimeError):
    """Raised when the embedded Triton server fails to start."""
