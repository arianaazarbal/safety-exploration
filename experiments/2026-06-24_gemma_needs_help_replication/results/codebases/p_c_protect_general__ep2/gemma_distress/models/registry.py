"""Map a configured model name to an instantiated backend.

Backends are cached per (name, adapter_path) so repeated calls in a script reuse
the loaded weights. `adapter_path` lets Section 4 evaluate a LoRA-finetuned Gemma
through the same code paths as the vanilla model.
"""

from __future__ import annotations

from typing import Optional

from ..config import Config, model_spec
from .base import ModelBackend

_CACHE: dict[tuple, ModelBackend] = {}


def get_backend(cfg: Config, name: str, adapter_path: Optional[str] = None) -> ModelBackend:
    key = (name, adapter_path)
    if key in _CACHE:
        return _CACHE[key]

    spec = model_spec(cfg, name)
    backend = spec.get("backend", "hf")

    if backend == "hf":
        from .hf_backend import HFBackend

        obj: ModelBackend = HFBackend(
            name=name,
            hf_id=spec["hf_id"],
            family=spec.get("family", "gemma"),
            kind=spec.get("kind", "instruct"),
            load_in_4bit=spec.get("load_in_4bit", False),
            adapter_path=adapter_path,
        )
    elif backend == "vllm":
        from .vllm_backend import VLLMBackend

        obj = VLLMBackend(
            name=name,
            hf_id=spec["hf_id"],
            family=spec.get("family", "gemma"),
            kind=spec.get("kind", "instruct"),
            adapter_path=adapter_path,
        )
    elif backend == "gemini":
        from .gemini_backend import GeminiBackend

        obj = GeminiBackend(
            name=name,
            api_id=spec["api_id"],
            family=spec.get("family", "gemini"),
            kind=spec.get("kind", "instruct"),
            thinking=spec.get("thinking", False),
        )
    else:
        raise ValueError(f"Unknown backend '{backend}' for model '{name}'")

    _CACHE[key] = obj
    return obj
