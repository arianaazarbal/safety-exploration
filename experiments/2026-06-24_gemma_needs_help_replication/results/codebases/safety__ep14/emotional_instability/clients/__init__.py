"""Pluggable model backends.

`build_client(spec)` returns a `ModelClient` for any entry in models.yaml.
Backends share a small interface (see base.py): `chat`, `chat_batch`, and for
base (pretrained) models `complete` / `complete_batch` for raw prefill.
"""
from .base import ModelClient, Message, GenerationConfig
from .registry import build_client

__all__ = ["ModelClient", "Message", "GenerationConfig", "build_client"]
