from __future__ import annotations

from typing import Any, Dict, List, Literal

from pydantic import BaseModel, field_validator

TruncationMode = Literal["NONE", "END", "START"]

QUERY_INPUT_NAME = "QUERY"
PASSAGES_INPUT_NAME = "PASSAGES"
TRUNCATE_INPUT_NAME = "TRUNCATE"
SCORES_OUTPUT_NAME = "SCORES"


class TextBlock(BaseModel):
    """
    Basic text container used across request fields.
    """

    text: str

    model_config = {"populate_by_name": True, "extra": "ignore"}

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        """
        Ensure text fields are non-empty and trimmed.
        """
        normalized = value.strip()
        if not normalized:
            raise ValueError("text must not be empty")
        return normalized


class Passage(TextBlock):
    """
    Passage payload containing text and optional metadata.
    """

    metadata: Dict[str, Any] | None = None


class RankingRequest(BaseModel):
    """
    HTTP ranking request payload.
    """

    model: str | None = None
    query: TextBlock
    passages: List[Passage]
    truncate: TruncationMode = "END"

    model_config = {"populate_by_name": True, "extra": "ignore"}

    @field_validator("passages")
    @classmethod
    def validate_passages(cls, value: List[Passage]) -> List[Passage]:
        """
        Guard against empty passage lists.
        """
        if not value:
            raise ValueError("passages must contain at least one entry")
        return value
