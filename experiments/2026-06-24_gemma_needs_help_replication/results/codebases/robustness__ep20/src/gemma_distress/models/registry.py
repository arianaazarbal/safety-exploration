"""Maps short model names to concrete backends.

Identifiers follow the paper (Appendix B.1):
  * Gemma instruct/base:  google/gemma-3-{27b,12b}-{it,pt}
  * Gemini (OpenRouter):  google/gemini-2.5-{flash,pro}

Open-weight models are instantiated lazily (they load many GB of weights), so
the registry stores *factories*, not live objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .base import ChatModel


@dataclass(frozen=True)
class ModelSpec:
    name: str
    backend: str                 # "hf" | "openrouter"
    identifier: str              # HF id or OpenRouter id
    is_base_model: bool = False
    family: str = ""             # "gemma" | "gemini"


REGISTRY: dict[str, ModelSpec] = {
    # Gemma — open weights (local HF inference)
    "gemma-3-27b-it": ModelSpec("gemma-3-27b-it", "hf", "google/gemma-3-27b-it", family="gemma"),
    "gemma-3-12b-it": ModelSpec("gemma-3-12b-it", "hf", "google/gemma-3-12b-it", family="gemma"),
    "gemma-3-27b-pt": ModelSpec("gemma-3-27b-pt", "hf", "google/gemma-3-27b-pt",
                                is_base_model=True, family="gemma"),
    "gemma-3-12b-pt": ModelSpec("gemma-3-12b-pt", "hf", "google/gemma-3-12b-pt",
                                is_base_model=True, family="gemma"),
    # Gemini — closed weights (OpenRouter API)
    "gemini-2.5-flash": ModelSpec("gemini-2.5-flash", "openrouter",
                                  "google/gemini-2.5-flash", family="gemini"),
    "gemini-2.5-pro": ModelSpec("gemini-2.5-pro", "openrouter",
                                "google/gemini-2.5-pro", family="gemini"),
}


def build_model(name: str, *, adapter_path: str | None = None, **kw) -> ChatModel:
    """Instantiate a model by registry name. `adapter_path` attaches a LoRA
    adapter (used to evaluate the DPO/SFT finetunes); only valid for HF models."""
    if name not in REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Known: {sorted(REGISTRY)}")
    spec = REGISTRY[name]
    if spec.backend == "hf":
        from .hf import HFChatModel
        return HFChatModel(spec.name, spec.identifier, adapter_path=adapter_path,
                           is_base_model=spec.is_base_model, **kw)
    elif spec.backend == "openrouter":
        if adapter_path:
            raise ValueError("Cannot attach a LoRA adapter to an API model.")
        from .openrouter import OpenRouterChatModel
        return OpenRouterChatModel(spec.name, spec.identifier, **kw)
    raise ValueError(f"Unknown backend '{spec.backend}'")


def model_factory(name: str, **kw) -> Callable[[], ChatModel]:
    return lambda: build_model(name, **kw)
