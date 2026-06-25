"""Model client abstractions.

All experiments talk to models through the small :class:`ModelClient` interface
in ``base.py`` so the same rollout / judging code drives a locally-hosted Gemma,
a Gemini endpoint over OpenRouter, or an Anthropic judge without change.

Factory: :func:`build_client` maps a config key (e.g. ``"gemma-3-27b-it"`` or
``"gemini-2.5-flash"``) to the appropriate backend.
"""
from .base import Message, ModelClient
from .factory import build_client

__all__ = ["Message", "ModelClient", "build_client"]
