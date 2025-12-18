from __future__ import annotations

from oim_common.errors import InvalidImageError


class ModelLoadError(RuntimeError):
    """
    Raised when the OCR model or its dependencies cannot be loaded.
    """


__all__ = ["ModelLoadError", "InvalidImageError"]
