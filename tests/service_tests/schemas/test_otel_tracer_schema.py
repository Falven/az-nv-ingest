# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nv_ingest.framework.schemas.framework_otel_tracer_schema import OpenTelemetryTracerSchema


def test_otel_tracer_schema_defaults():
    schema = OpenTelemetryTracerSchema()
    assert schema.otel_endpoint == "localhost:4317"
    assert schema.otel_headers == {}
    assert schema.resolved_connection_string is None
    assert schema.raise_on_failure is False, "Default value for raise_on_failure should be False."


def test_otel_tracer_schema_custom_values():
    schema = OpenTelemetryTracerSchema(
        raise_on_failure=True,
        otel_endpoint="https://example:4317",
        otel_headers={"authorization": "Bearer token"},
        app_insights_instrumentation_key="abc-123",
    )
    assert schema.raise_on_failure is True, "Custom value for raise_on_failure should be respected."
    assert schema.otel_headers == {"authorization": "Bearer token"}
    assert schema.resolved_connection_string == "InstrumentationKey=abc-123"
