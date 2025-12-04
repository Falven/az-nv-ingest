from __future__ import annotations

import os
from enum import Enum
from typing import Any
from typing import Dict
from typing import List

from fastapi import Body
from fastapi import FastAPI
from pydantic import BaseModel
from pydantic import Field

from open_shims.constants import GRAPHIC_BOXES
from open_shims.constants import OCR_TEXT_DETECTIONS
from open_shims.constants import PAGE_BOXES
from open_shims.constants import PARSE_ARGUMENTS
from open_shims.constants import TABLE_BOXES
from open_shims.constants import build_bounding_box_payload
from open_shims.constants import build_ocr_payload
from open_shims.constants import build_parse_payload
from open_shims.constants import sample_chat_completion_request
from open_shims.constants import sample_infer_request


class ShimKind(str, Enum):
    PAGE_ELEMENTS = "page-elements"
    GRAPHIC_ELEMENTS = "graphic-elements"
    TABLE_STRUCTURE = "table-structure"
    OCR = "ocr"
    PARSE = "nemotron-parse"


class ImageUrl(BaseModel):
    type: str = Field(..., description="The input type; should be 'image_url'.")
    url: str = Field(..., description="Data URL for the image payload.")

    model_config = {"extra": "allow"}


class InferRequest(BaseModel):
    input: List[ImageUrl]

    model_config = {"extra": "allow"}


class ChatContent(BaseModel):
    type: str
    image_url: Dict[str, str] = Field(..., alias="image_url")

    model_config = {"extra": "allow"}


class ChatMessage(BaseModel):
    role: str
    content: List[ChatContent]

    model_config = {"extra": "allow"}


class ChatCompletionsRequest(BaseModel):
    model: str
    messages: List[ChatMessage]

    model_config = {"extra": "allow"}


def get_kind() -> ShimKind:
    requested = os.getenv("SHIM_KIND", ShimKind.PAGE_ELEMENTS.value)
    try:
        return ShimKind(requested)
    except ValueError:
        return ShimKind.PAGE_ELEMENTS


def create_app(kind: ShimKind | None = None) -> FastAPI:
    service_kind = kind or get_kind()
    app = FastAPI(title=f"{service_kind.value} shim", version="0.1.0")

    @app.get("/health")
    @app.get("/v1/health/ready")
    async def health() -> Dict[str, Any]:
        return {"status": "ok", "service": service_kind.value}

    @app.get("/sample")
    async def sample() -> Dict[str, Any]:
        if service_kind == ShimKind.PARSE:
            return sample_chat_completion_request()
        return sample_infer_request()

    if service_kind == ShimKind.PARSE:

        @app.post("/v1/chat/completions")
        async def chat_completions(request: ChatCompletionsRequest = Body(...)) -> Dict[str, Any]:
            _ = request  # request is validated for shape; content is ignored for determinism.
            return build_parse_payload(PARSE_ARGUMENTS)

    else:

        @app.post("/v1/infer")
        async def infer(request: InferRequest = Body(...)) -> Dict[str, Any]:
            batch_size = len(request.input)

            if service_kind == ShimKind.OCR:
                return build_ocr_payload(OCR_TEXT_DETECTIONS, batch_size)

            if service_kind == ShimKind.PAGE_ELEMENTS:
                return build_bounding_box_payload(PAGE_BOXES, batch_size)

            if service_kind == ShimKind.GRAPHIC_ELEMENTS:
                return build_bounding_box_payload(GRAPHIC_BOXES, batch_size)

            if service_kind == ShimKind.TABLE_STRUCTURE:
                return build_bounding_box_payload(TABLE_BOXES, batch_size)

            return build_bounding_box_payload(PAGE_BOXES, batch_size)

    return app


__all__ = ["ShimKind", "create_app", "sample_infer_request", "sample_chat_completion_request", "SAMPLE_IMAGE_URL"]
