"""Provider adapters. The Anthropic adapter is the reference implementation;
others are honest stubs to be filled in with each provider's official SDK."""

from .base import ModelClient, build_client

__all__ = ["ModelClient", "build_client"]
