from __future__ import annotations


class InvalidImageError(Exception):
    """Raised when an input image payload cannot be decoded."""


class InferenceError(Exception):
    """Raised when model inference fails unexpectedly."""


class TritonStartupError(Exception):
    """Raised when the embedded Triton server cannot start or become ready."""


class TritonInferenceError(Exception):
    """Raised when Triton inference fails or returns malformed data."""
