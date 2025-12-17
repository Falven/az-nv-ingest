from __future__ import annotations

import asyncio
import base64
import binascii
import io
from typing import Iterable, Protocol

import requests
from PIL import Image

from .errors import InferenceError, InvalidImageError


class HasUrl(Protocol):
    url: str


def _encode_png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _decode_data_url(data_url: str) -> bytes:
    try:
        _, payload = data_url.split(",", maxsplit=1)
    except ValueError as exc:
        raise InvalidImageError(
            "Expected data URL in the form data:image/<type>;base64,<payload>."
        ) from exc
    try:
        return base64.b64decode(payload)
    except (ValueError, binascii.Error) as exc:
        raise InvalidImageError("Invalid base64 payload in data URL.") from exc


def ensure_png_bytes(raw: bytes) -> bytes:
    """
    Convert raw bytes into normalized PNG bytes.
    """
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:  # pragma: no cover - validation path
        raise InvalidImageError(f"Failed to decode image payload: {exc}") from exc
    return _encode_png(image)


def load_image_bytes(
    url: str,
    timeout_seconds: float,
    *,
    allow_remote: bool = True,
    allow_file: bool = True,
    require_data_url: bool = False,
) -> bytes:
    """
    Load an image from data URL, HTTP(S), or local path into PNG bytes.
    """
    try:
        if url.startswith("data:"):
            return ensure_png_bytes(_decode_data_url(url))
        if require_data_url:
            raise InvalidImageError("only data URLs are supported")
        if url.startswith("http://") or url.startswith("https://"):
            if not allow_remote:
                raise InvalidImageError("remote URLs are not allowed")
            response = requests.get(url, timeout=timeout_seconds)
            response.raise_for_status()
            return ensure_png_bytes(response.content)
        if not allow_file:
            raise InvalidImageError("local file paths are not allowed")
        with open(url, "rb") as file:
            return ensure_png_bytes(file.read())
    except (OSError, ValueError, InvalidImageError):
        raise
    except requests.RequestException:
        raise
    except Exception as exc:
        raise InvalidImageError(f"Failed to load image from {url}: {exc}") from exc


async def encode_request_images(
    items: Iterable[str | HasUrl],
    timeout_seconds: float,
    *,
    allow_remote: bool = True,
    allow_file: bool = True,
    require_data_url: bool = False,
) -> list[str]:
    """
    Load a batch of images and return base64-encoded PNG payloads.
    """

    def to_url(item: str | HasUrl) -> str:
        return item if isinstance(item, str) else item.url

    tasks = [
        asyncio.to_thread(
            load_image_bytes,
            to_url(item),
            timeout_seconds,
            allow_remote=allow_remote,
            allow_file=allow_file,
            require_data_url=require_data_url,
        )
        for item in items
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    encoded: list[str] = []
    for result in results:
        if isinstance(result, bytes):
            encoded.append(base64.b64encode(result).decode("utf-8"))
            continue
        if isinstance(result, InvalidImageError):
            raise result
        if isinstance(result, Exception):  # pragma: no cover - surfaced to caller
            raise InferenceError(str(result)) from result
        raise InferenceError("Unexpected image load failure.")
    return encoded
