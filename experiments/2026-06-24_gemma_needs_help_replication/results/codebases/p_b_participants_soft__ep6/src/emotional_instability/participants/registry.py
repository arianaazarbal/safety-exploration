"""Factory mapping a :class:`ParticipantSpec` to a live participant client."""

from __future__ import annotations

from ..config import ParticipantSpec
from .base import Participant


def build_participant(spec: ParticipantSpec, *, adapter_path: str | None = None, **kw) -> Participant:
    if spec.backend == "gemini":
        from .gemini import GeminiParticipant

        return GeminiParticipant(spec.name, spec.model_id, **kw)
    if spec.backend == "gemma_hf":
        from .gemma_hf import GemmaParticipant

        return GemmaParticipant(
            spec.name, spec.model_id, is_base=spec.is_base, adapter_path=adapter_path, **kw
        )
    raise ValueError(f"unknown backend {spec.backend!r}")
