from __future__ import annotations


class ModelLoadError(RuntimeError):
    """
    Raised when the OCR model or its dependencies cannot be loaded.
    """


class InvalidImageError(ValueError):
    """
    Raised when an image payload cannot be decoded.
    """
