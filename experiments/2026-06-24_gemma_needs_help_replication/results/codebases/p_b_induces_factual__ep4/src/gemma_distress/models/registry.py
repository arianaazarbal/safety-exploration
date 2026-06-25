"""Model registry — Gemma + Gemini only (replication scope).

``TARGET_MODELS`` are the models scored in the Section 2 evals (Figure 1/2).
Base models and the DPO/SFT variants are constructed on demand by the
Section 3 / Section 4 scripts, not listed here.
"""
from __future__ import annotations

from dataclasses import dataclass

from .base import ChatModel


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str            # "gemma" | "gemini"
    backend: str           # "gemini" | "gemma_api" | "gemma_local"
    hf_id: str | None = None
    is_base_model: bool = False


# Section 2 targets within scope. Gemma can be served via the hosted API
# (backend="gemma_api") or locally (backend="gemma_local"); we default the eval
# to the API for convenience and switch to local for Section 3/4.
TARGET_MODELS: dict[str, ModelSpec] = {
    "gemma-3-27b-it": ModelSpec(
        "gemma-3-27b-it", "gemma", "gemma_api", hf_id="google/gemma-3-27b-it"
    ),
    "gemma-3-12b-it": ModelSpec(
        "gemma-3-12b-it", "gemma", "gemma_api", hf_id="google/gemma-3-12b-it"
    ),
    "gemini-2.5-flash": ModelSpec("gemini-2.5-flash", "gemini", "gemini"),
    "gemini-2.5-pro": ModelSpec("gemini-2.5-pro", "gemini", "gemini"),
}

# Base / continuation models used only in the Section 3 prefilling study. Of the
# families the paper compares (Gemma, Qwen, OLMo) only Gemma is in scope.
PREFILL_MODELS: dict[str, ModelSpec] = {
    "gemma-3-27b-pt": ModelSpec(
        "gemma-3-27b-pt", "gemma", "gemma_local",
        hf_id="google/gemma-3-27b-pt", is_base_model=True,
    ),
    "gemma-3-27b-it": ModelSpec(
        "gemma-3-27b-it", "gemma", "gemma_local", hf_id="google/gemma-3-27b-it"
    ),
}


def list_targets() -> list[str]:
    return list(TARGET_MODELS)


def get_model(
    name: str,
    *,
    backend: str | None = None,
    adapter_path: str | None = None,
    load_in_4bit: bool = False,
    spec: ModelSpec | None = None,
) -> ChatModel:
    """Instantiate a model by name.

    ``backend`` overrides the registry default (e.g. force a Gemma target to run
    locally). ``adapter_path`` loads a LoRA adapter (local backend only).
    """
    spec = spec or TARGET_MODELS.get(name) or PREFILL_MODELS.get(name)
    if spec is None:
        raise KeyError(f"Unknown model {name!r}; in-scope models: {list_targets()}")
    backend = backend or spec.backend

    if backend == "gemini":
        from .gemini import GeminiModel

        return GeminiModel(spec.name)
    if backend == "gemma_api":
        from .gemma_api import GemmaAPIModel

        return GemmaAPIModel(spec.name)
    if backend == "gemma_local":
        from .gemma_local import GemmaLocalModel

        if spec.hf_id is None:
            raise ValueError(f"{name}: no hf_id for local backend")
        return GemmaLocalModel(
            spec.name,
            spec.hf_id,
            is_base_model=spec.is_base_model,
            adapter_path=adapter_path,
            load_in_4bit=load_in_4bit,
        )
    raise ValueError(f"Unknown backend {backend!r}")
