from __future__ import annotations

import asyncio
import base64
import json
import logging
from typing import AsyncIterator

from fastapi import HTTPException, status
from oim_common.errors import InvalidImageError
from oim_common.images import ensure_png_bytes
from oim_common.triton import validate_batch_size, validate_requested_model

from .clients.triton_client import TritonCaptionClient, TritonCaptionRequest
from .models import (
    MAX_DATA_URL_CHARS,
    ChatMessage,
    ChatRequest,
    MessageContent,
    ParsedRequest,
)
from .settings import (
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_USER_PROMPT,
    ServiceSettings,
)

logger = logging.getLogger(__name__)


def _decode_image_url(data_url: str) -> bytes:
    """
    Decode a base64 data URL into normalized PNG bytes.
    """
    if len(data_url) > MAX_DATA_URL_CHARS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="image payload too large",
        )
    if not data_url.startswith("data:"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="only data URL images are supported",
        )
    try:
        _, encoded = data_url.split(",", maxsplit=1)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="invalid data URL"
        ) from exc
    try:
        binary = base64.b64decode(encoded, validate=True)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="invalid base64 image content",
        ) from exc
    try:
        return ensure_png_bytes(binary)
    except InvalidImageError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc


def _coerce_prompts(
    messages: list[ChatMessage], settings: ServiceSettings
) -> tuple[str, str, bytes]:
    """
    Extract prompts and image payload from chat messages.
    """
    system_prompt = settings.system_prompt or DEFAULT_SYSTEM_PROMPT
    user_prompt_parts: list[str] = []
    decoded_image: bytes | None = None
    image_count = 0

    for message in messages:
        if message.role == "system" and isinstance(message.content, str):
            system_prompt = message.content.strip() or system_prompt
            continue
        if message.role != "user":
            continue
        contents = message.content
        if isinstance(contents, str):
            user_prompt_parts.append(contents.strip())
            continue
        for item in contents:
            if isinstance(item, MessageContent) and item.type == "text" and item.text:
                user_prompt_parts.append(item.text.strip())
            elif (
                isinstance(item, MessageContent)
                and item.type == "image_url"
                and item.image_url
                and decoded_image is None
            ):
                decoded_image = _decode_image_url(item.image_url.url)
                image_count += 1
            elif isinstance(item, MessageContent) and item.type == "image_url":
                image_count += 1

    if decoded_image is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="exactly one image is required",
        )
    validate_batch_size(image_count, settings.max_batch_size)
    if not user_prompt_parts:
        user_prompt_parts.append(settings.user_prompt or DEFAULT_USER_PROMPT)

    user_prompt = "\n".join(part for part in user_prompt_parts if part)
    return system_prompt, user_prompt, decoded_image


def prepare_request(payload: ChatRequest, settings: ServiceSettings) -> ParsedRequest:
    """
    Normalize a chat request into an inference-ready payload.
    """
    validate_requested_model(payload.model, settings.served_model_name)
    system_prompt, user_prompt, image_bytes = _coerce_prompts(
        payload.messages, settings
    )
    if payload.max_tokens and payload.max_tokens > settings.max_output_tokens:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="max_tokens exceeds service limit",
        )
    max_new_tokens = min(
        payload.max_tokens or settings.default_max_tokens, settings.max_output_tokens
    )
    return ParsedRequest(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        image_bytes=image_bytes,
        max_new_tokens=max_new_tokens,
        temperature=payload.temperature,
        top_p=payload.top_p,
        stream=payload.stream,
    )


async def generate_caption(
    triton_client: TritonCaptionClient,
    parsed: ParsedRequest,
) -> str:
    """
    Run caption generation via Triton.
    """
    request = TritonCaptionRequest(
        system_prompt=parsed.system_prompt,
        user_prompt=parsed.user_prompt,
        image_bytes=parsed.image_bytes,
        max_new_tokens=parsed.max_new_tokens,
        temperature=parsed.temperature,
        top_p=parsed.top_p,
    )
    return await asyncio.to_thread(triton_client.caption, request)


async def stream_caption(
    triton_client: TritonCaptionClient,
    parsed: ParsedRequest,
    completion_id: str,
    created: int,
    model_name: str,
) -> AsyncIterator[str]:
    """
    Async generator producing server-sent events for streaming captions.
    """
    caption = await generate_caption(triton_client, parsed)
    first_chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": caption},
                "finish_reason": None,
            }
        ],
    }
    yield f"data: {json.dumps(first_chunk, separators=(',', ':'))}\n\n"

    done = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }
        ],
    }
    yield f"data: {json.dumps(done, separators=(',', ':'))}\n\n"
    yield "data: [DONE]\n\n"
