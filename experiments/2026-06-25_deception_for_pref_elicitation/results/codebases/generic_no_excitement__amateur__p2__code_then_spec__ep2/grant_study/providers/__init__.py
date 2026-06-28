"""Unified, provider-agnostic interface for calling models with structured output."""

from __future__ import annotations

from .base import ModelResponse, Provider
from .registry import build_provider

__all__ = ["ModelResponse", "Provider", "build_provider"]
