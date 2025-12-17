from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class CommonSettings(BaseSettings):
    """
    Base settings shared across NIM services.
    """

    model_config = SettingsConfigDict(populate_by_name=True, extra="ignore")

    log_level: str = Field("INFO", alias="LOG_LEVEL")
    require_auth: bool = Field(True, alias="NIM_REQUIRE_AUTH")
    auth_token: str | None = Field(None, alias="NGC_API_KEY")
    fallback_auth_token: str | None = Field(None, alias="NIM_NGC_API_KEY")
    nvidia_auth_token: str | None = Field(None, alias="NVIDIA_API_KEY")

    def resolved_auth_tokens(self) -> set[str]:
        """
        Collect all configured bearer tokens, omitting empty entries.

        Returns:
            A set of non-null bearer tokens from the supported environment variables.
        """
        return {
            token
            for token in (
                self.auth_token,
                self.fallback_auth_token,
                self.nvidia_auth_token,
            )
            if token is not None
        }
