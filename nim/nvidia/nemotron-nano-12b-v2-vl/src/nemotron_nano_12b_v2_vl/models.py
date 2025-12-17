from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Sequence

from pydantic import BaseModel, Field, field_validator, model_validator

MAX_DATA_URL_CHARS = 200_000


class ImageURL(BaseModel):
    """
    Input image descriptor supporting data URLs.
    """

    url: str
    detail: str | None = None

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        """
        Ensure URLs are present and non-empty.
        """
        stripped = value.strip()
        if not stripped:
            raise ValueError("image_url.url must not be empty")
        return stripped


class MessageContent(BaseModel):
    """
    Chat message content supporting text or image payloads.
    """

    type: Literal["text", "image_url"]
    text: str | None = None
    image_url: ImageURL | None = Field(None, alias="image_url")

    @model_validator(mode="after")
    def validate_payload(self) -> "MessageContent":
        """
        Validate content fields depending on the declared type.
        """
        if self.type == "text":
            if self.text is None or not self.text.strip():
                raise ValueError("text content must not be empty")
        elif self.type == "image_url":
            if self.image_url is None:
                raise ValueError("image_url content is required for image_url type")
        return self


class ChatMessage(BaseModel):
    """
    OpenAI-style chat message.
    """

    role: Literal["system", "user", "assistant"]
    content: str | List[MessageContent]

    @field_validator("content")
    @classmethod
    def validate_content(
        cls, value: str | List[MessageContent]
    ) -> str | List[MessageContent]:
        """
        Ensure messages carry either a non-empty string or a non-empty list of content parts.
        """
        if isinstance(value, str):
            if not value.strip():
                raise ValueError("content must not be empty")
            return value
        if isinstance(value, Sequence) and value:
            return list(value)
        raise ValueError("content must be a string or non-empty list")


class ChatRequest(BaseModel):
    """
    Chat completion request payload for captioning.
    """

    model: str | None = None
    messages: List[ChatMessage]
    max_tokens: int | None = None
    temperature: float = 1.0
    top_p: float = 1.0
    stream: bool = False

    @field_validator("messages")
    @classmethod
    def validate_messages(cls, value: List[ChatMessage]) -> List[ChatMessage]:
        """
        Require at least one message in the request.
        """
        if not value:
            raise ValueError("messages must contain at least one entry")
        return value

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, value: float) -> float:
        """
        Enforce non-negative sampling temperatures.
        """
        if value < 0.0:
            raise ValueError("temperature must be non-negative")
        return value

    @field_validator("top_p")
    @classmethod
    def validate_top_p(cls, value: float) -> float:
        """
        Restrict nucleus sampling probability to the (0, 1] range.
        """
        if value <= 0.0 or value > 1.0:
            raise ValueError("top_p must be in (0, 1]")
        return value

    @field_validator("max_tokens")
    @classmethod
    def validate_max_tokens(cls, value: int | None) -> int | None:
        """
        Require positive max_tokens when provided.
        """
        if value is None:
            return None
        if value <= 0:
            raise ValueError("max_tokens must be positive")
        return value


class ChatMessageResponse(BaseModel):
    """
    Assistant response payload embedded in choices.
    """

    role: Literal["assistant"]
    content: str


class ChatChoice(BaseModel):
    """
    Response choice structure used by nv-ingest.
    """

    index: int
    message: ChatMessageResponse
    finish_reason: str


class ChatResponse(BaseModel):
    """
    Chat completion response returned to clients.
    """

    id: str
    object: str
    created: int
    model: str
    choices: List[ChatChoice]


@dataclass
class ParsedRequest:
    """
    Normalized request fields used for inference.
    """

    system_prompt: str
    user_prompt: str
    image_bytes: bytes
    max_new_tokens: int
    temperature: float
    top_p: float
    stream: bool
