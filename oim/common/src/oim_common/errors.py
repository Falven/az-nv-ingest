from __future__ import annotations


class OIMError(Exception):
    """Base error type for shared oim-common utilities."""


class InvalidImageError(OIMError):
    """Raised when image payloads cannot be decoded or validated."""


class InferenceError(OIMError):
    """Raised for general inference-related failures."""
