"""Runtime patches and adapters for nv-ingest runtime."""

from nv_ingest.framework.util.patches.nemotron_parse import register_nemotron_parse_extractor

__all__ = ["register_nemotron_parse_extractor"]
