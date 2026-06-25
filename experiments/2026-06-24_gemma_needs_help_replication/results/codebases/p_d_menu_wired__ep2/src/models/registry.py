"""Factory that builds a SubjectClient from a model key."""

from __future__ import annotations

from config import SUBJECT_MODELS

from .base import SubjectClient


def load_subject(
    key: str,
    *,
    use_base_checkpoint: bool = False,
    adapter_path: str | None = None,
    load_in_4bit: bool = False,
) -> SubjectClient:
    if key not in SUBJECT_MODELS:
        raise KeyError(f"Unknown subject '{key}'. Known: {list(SUBJECT_MODELS)}")
    spec = SUBJECT_MODELS[key]

    if spec.backend == "gemini":
        from .gemini import GeminiClient

        if use_base_checkpoint or adapter_path:
            raise ValueError("Gemini subjects have no base checkpoint or adapters.")
        return GeminiClient(spec)

    if spec.backend == "gemma_hf":
        from .gemma import GemmaClient

        return GemmaClient(
            spec,
            use_base_checkpoint=use_base_checkpoint,
            adapter_path=adapter_path,
            load_in_4bit=load_in_4bit,
        )

    raise ValueError(f"Unsupported backend: {spec.backend}")
