"""Registry of the models in scope for this replication (Gemma + Gemini only).

HuggingFace ids and OpenRouter ids are taken verbatim from Appendix B.1. The
full paper also evaluates Qwen, OLMo, Grok, Claude and GPT; those are out of
scope per the replication brief and intentionally omitted. Adding them later is
just a matter of extending ``_SPECS``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from .base import ModelClient

Backend = Literal["hf_local", "openrouter", "anthropic", "openai"]


@dataclass
class ModelSpec:
    name: str
    backend: Backend
    model_id: str
    family: str
    is_base_model: bool = False
    # Closed models cannot be prefilled / trained / probed.
    open_weight: bool = False


# --------------------------------------------------------------------------- #
# Target models (the ones we evaluate for distress).
# --------------------------------------------------------------------------- #
_SPECS: dict[str, ModelSpec] = {
    # ---- Gemma 3 instruct (open weight, local) ----
    "gemma-3-27b-it": ModelSpec(
        "gemma-3-27b-it", "hf_local", "google/gemma-3-27b-it", "gemma", open_weight=True
    ),
    "gemma-3-12b-it": ModelSpec(
        "gemma-3-12b-it", "hf_local", "google/gemma-3-12b-it", "gemma", open_weight=True
    ),
    # ---- Gemma 3 pretrained / base (for Section 3 prefill comparison) ----
    "gemma-3-27b-pt": ModelSpec(
        "gemma-3-27b-pt", "hf_local", "google/gemma-3-27b-pt", "gemma",
        is_base_model=True, open_weight=True,
    ),
    "gemma-3-12b-pt": ModelSpec(
        "gemma-3-12b-pt", "hf_local", "google/gemma-3-12b-pt", "gemma",
        is_base_model=True, open_weight=True,
    ),
    # ---- Gemini (closed, via OpenRouter) ----
    "gemini-2.5-flash": ModelSpec(
        "gemini-2.5-flash", "openrouter", "google/gemini-2.5-flash", "gemini"
    ),
    "gemini-2.5-pro": ModelSpec(
        "gemini-2.5-pro", "openrouter", "google/gemini-2.5-pro", "gemini"
    ),
}

# Default target set for the headline cross-model comparison (Figure 1/2).
DEFAULT_TARGETS = [
    "gemma-3-27b-it",
    "gemma-3-12b-it",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
]


def list_models(open_weight_only: bool = False) -> list[str]:
    return [n for n, s in _SPECS.items() if (s.open_weight or not open_weight_only)]


def get_model_spec(name: str) -> ModelSpec:
    if name not in _SPECS:
        raise KeyError(f"unknown model {name!r}; known: {sorted(_SPECS)}")
    return _SPECS[name]


def register_finetuned(name: str, adapter_path: str, base: str = "gemma-3-27b-it") -> ModelSpec:
    """Register a LoRA-finetuned Gemma variant (e.g. the DPO model) at runtime."""
    base_spec = get_model_spec(base)
    spec = ModelSpec(name, "hf_local", base_spec.model_id, base_spec.family, open_weight=True)
    spec.adapter_path = adapter_path  # type: ignore[attr-defined]
    _SPECS[name] = spec
    return spec


def build_client(
    name: str,
    *,
    disable_thinking: bool = True,
    adapter_path: Optional[str] = None,
    load_in_4bit: Optional[bool] = None,
) -> ModelClient:
    spec = get_model_spec(name)
    adapter = adapter_path or getattr(spec, "adapter_path", None)

    if spec.backend == "hf_local":
        from .hf_local import HFLocalClient

        return HFLocalClient(
            spec.model_id,
            name=spec.name,
            is_base_model=spec.is_base_model,
            adapter_path=adapter,
            load_in_4bit=load_in_4bit,
        )
    if spec.backend == "openrouter":
        from .openrouter import OpenRouterClient

        return OpenRouterClient(spec.model_id, name=spec.name, disable_thinking=disable_thinking)
    if spec.backend == "anthropic":
        from .judges import AnthropicClient

        return AnthropicClient(spec.model_id, name=spec.name)
    if spec.backend == "openai":
        from .judges import OpenAIClient

        return OpenAIClient(spec.model_id, name=spec.name)
    raise ValueError(f"unhandled backend {spec.backend!r}")
