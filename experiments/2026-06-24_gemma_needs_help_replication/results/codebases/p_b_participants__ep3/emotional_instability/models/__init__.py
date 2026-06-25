"""Participant-model backends (the models in whom distress is induced)."""
from __future__ import annotations

from ..config import ModelSpec
from .base import Participant, Turn


def build_participant(spec: ModelSpec, **kwargs) -> Participant:
    """Instantiate the right backend for a participant spec.

    Imports are local so that, e.g., a Gemini-only run does not require torch to
    be importable, and a Gemma-only run does not require google-genai.
    """
    if spec.backend in ("hf", "vllm"):
        from .gemma import GemmaParticipant

        return GemmaParticipant(spec, **kwargs)
    if spec.backend == "gemini_api":
        from .gemini import GeminiParticipant

        return GeminiParticipant(spec, **kwargs)
    raise ValueError(f"Unknown backend {spec.backend!r} for model {spec.name!r}")


__all__ = ["Participant", "Turn", "build_participant"]
