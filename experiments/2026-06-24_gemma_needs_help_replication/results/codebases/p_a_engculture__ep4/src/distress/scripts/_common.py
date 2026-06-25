"""Helpers shared across CLI scripts: run-dir layout and provider loading
(optionally with a trained LoRA adapter for local Gemma)."""

from __future__ import annotations

from pathlib import Path

from ..config import RUN_DIR, get_model
from ..models import ModelProvider, load_provider


def out_dir(name: str) -> Path:
    d = RUN_DIR / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def make_provider(
    subject_key: str,
    *,
    adapter_path: str | None = None,
    backend: str | None = None,
    use_cache: bool = True,
) -> ModelProvider:
    """Load a provider for ``subject_key``.

    ``backend`` can override the spec's provider (e.g. force 'vllm' for a local
    Gemma sweep). ``adapter_path`` attaches a trained LoRA adapter (local only).
    """
    spec = get_model(subject_key)
    if backend and backend != spec.provider:
        import dataclasses

        spec = dataclasses.replace(spec, provider=backend)
    kwargs: dict = {}
    if spec.provider in {"hf", "vllm"} and adapter_path:
        kwargs["adapter_path"] = adapter_path
    if spec.provider in {"openrouter", "anthropic", "openai"}:
        kwargs["use_cache"] = use_cache
    return load_provider(spec, **kwargs)
