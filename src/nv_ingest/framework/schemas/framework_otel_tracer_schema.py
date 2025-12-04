# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from typing import Dict, Optional

from pydantic import ConfigDict, BaseModel, Field


class OpenTelemetryTracerSchema(BaseModel):
    otel_endpoint: str = "localhost:4317"
    otel_headers: Dict[str, str] = Field(default_factory=dict)
    app_insights_connection_string: Optional[str] = None
    app_insights_instrumentation_key: Optional[str] = None
    raise_on_failure: bool = False
    model_config = ConfigDict(extra="forbid")

    @property
    def resolved_connection_string(self) -> Optional[str]:
        if self.app_insights_connection_string:
            return self.app_insights_connection_string
        if self.app_insights_instrumentation_key:
            return f"InstrumentationKey={self.app_insights_instrumentation_key}"
        return None
