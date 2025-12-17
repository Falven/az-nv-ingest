"""
Stubbed tritonclient package used for contract tests without a running Triton server.
"""

from .http import InferenceServerClient, InferInput, InferRequestedOutput  # noqa: F401
from .utils import InferenceServerException  # noqa: F401

__all__ = [
    "InferenceServerClient",
    "InferInput",
    "InferRequestedOutput",
    "InferenceServerException",
]
