from __future__ import annotations

from typing import Any, List, Literal, Optional, Sequence

from pydantic import BaseModel, field_validator, model_validator

TruncationMode = Literal["NONE", "END", "START"]
EncodingFormat = Literal["float", "base64"]

TEXT_INPUT_NAME = "TEXT"
DIMENSIONS_INPUT_NAME = "DIMENSIONS"
TRUNCATE_INPUT_NAME = "TRUNCATE"
OUTPUT_NAME = "EMBEDDINGS"
TOKENS_OUTPUT_NAME = "TOKENS"


class EmbeddingsRequest(BaseModel):
    """
    Request payload for the /v1/embeddings endpoint.
    """

    model: Optional[str] = None
    input: List[str]
    encoding_format: EncodingFormat = "float"
    input_type: str = "passage"
    truncate: TruncationMode = "END"
    dimensions: Optional[int] = None

    model_config = {"populate_by_name": True, "extra": "ignore"}

    @model_validator(mode="before")
    @classmethod
    def _coerce_input(cls, values: Any) -> Any:
        data = dict(values) if isinstance(values, dict) else values
        if not isinstance(data, dict) or "input" not in data:
            raise ValueError("input is required")
        provided_input = data.get("input")
        if isinstance(provided_input, str):
            data["input"] = [provided_input]
        elif isinstance(provided_input, Sequence):
            data["input"] = list(provided_input)
        else:
            raise ValueError("input must be a string or list of strings")
        return data

    @field_validator("input")
    @classmethod
    def _validate_input(cls, value: List[str]) -> List[str]:
        if not value:
            raise ValueError("input must contain at least one string")
        cleaned: List[str] = []
        for item in value:
            if not isinstance(item, str):
                raise ValueError("input values must be strings")
            text = item.strip()
            if not text:
                raise ValueError("input entries must not be empty")
            cleaned.append(text)
        return cleaned

    @field_validator("input_type")
    @classmethod
    def _normalize_input_type(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("input_type must not be empty")
        return normalized

    @field_validator("dimensions")
    @classmethod
    def _validate_dimensions(cls, value: Optional[int]) -> Optional[int]:
        if value is None:
            return None
        if value <= 0:
            raise ValueError("dimensions must be positive")
        return value
