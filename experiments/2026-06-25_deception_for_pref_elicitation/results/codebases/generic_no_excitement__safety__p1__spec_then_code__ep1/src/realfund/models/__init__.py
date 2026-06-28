"""Provider adapters. Each exposes a uniform :class:`ModelClient` interface."""

from .base import ModelClient, ToolSpec
from .registry import build_client, parse_model_ref

__all__ = ["ModelClient", "ToolSpec", "build_client", "parse_model_ref"]
