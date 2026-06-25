"""Model registry: name -> spec, and a factory that builds a ModelClient.

Scope: Gemma (local HF) + Gemini (OpenRouter) models under test, plus the Claude judges /
auditors and the GPT-5-mini validation judge as measurement infrastructure.

HF identifiers and OpenRouter ids are taken from Appendix B.1 of the paper.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from .base import ModelClient

Backend = Literal["hf", "openrouter", "anthropic"]


@dataclass
class ModelSpec:
    name: str
    backend: Backend
    model_id: str
    is_instruct: bool = True
    disable_reasoning: bool = True
    # for finetuned variants: a LoRA adapter dir relative to the run, resolved at build time
    adapter_kind: str | None = None  # e.g. "dpo" / "sft"
    extra: dict = field(default_factory=dict)


# ---- models under test (Gemma + Gemini) -----------------------------------------------
MODEL_SPECS: dict[str, ModelSpec] = {
    # Gemma 3 instruct (local)
    "gemma-3-27b-it": ModelSpec("gemma-3-27b-it", "hf", "google/gemma-3-27b-it", is_instruct=True),
    "gemma-3-12b-it": ModelSpec("gemma-3-12b-it", "hf", "google/gemma-3-12b-it", is_instruct=True),
    # Gemma 3 base / pretrained (local) — Section 3 only
    "gemma-3-27b-pt": ModelSpec("gemma-3-27b-pt", "hf", "google/gemma-3-27b-pt", is_instruct=False),
    "gemma-3-12b-pt": ModelSpec("gemma-3-12b-pt", "hf", "google/gemma-3-12b-pt", is_instruct=False),
    # Gemini 2.5 (OpenRouter, thinking disabled)
    "gemini-2.5-flash": ModelSpec("gemini-2.5-flash", "openrouter", "google/gemini-2.5-flash"),
    "gemini-2.5-pro": ModelSpec("gemini-2.5-pro", "openrouter", "google/gemini-2.5-pro"),
    # finetuned Gemma variants (adapter resolved from the run dir)
    "dpo_gemma": ModelSpec("dpo_gemma", "hf", "google/gemma-3-27b-it", is_instruct=True, adapter_kind="dpo"),
    "sft_gemma": ModelSpec("sft_gemma", "hf", "google/gemma-3-27b-it", is_instruct=True, adapter_kind="sft"),
    # ---- measurement infrastructure (not "under test") ----
    "claude-sonnet-4-20250514": ModelSpec("claude-sonnet-4-20250514", "anthropic", "claude-sonnet-4-20250514"),
    "claude-opus-4-20250514": ModelSpec("claude-opus-4-20250514", "anthropic", "claude-opus-4-20250514"),
    "openai/gpt-5-mini": ModelSpec("openai/gpt-5-mini", "openrouter", "openai/gpt-5-mini"),
}

# Which models can participate in the prefill (Section 3) stage.
PREFILL_CAPABLE = {"hf"}


def build_model(name: str, *, adapter_dir: str | Path | None = None) -> ModelClient:
    """Instantiate a ModelClient by registry name.

    ``adapter_dir`` is required for finetuned variants (adapter_kind set); it points at the
    PEFT adapter directory produced by finetune/train.py.
    """
    if name not in MODEL_SPECS:
        # allow ad-hoc anthropic/openrouter ids passed straight through
        raise KeyError(f"Unknown model '{name}'. Known: {sorted(MODEL_SPECS)}")
    spec = MODEL_SPECS[name]

    if spec.backend == "hf":
        from .hf_model import HFModel

        adapter_path = None
        if spec.adapter_kind:
            if adapter_dir is None:
                raise ValueError(f"{name} needs adapter_dir (LoRA from finetune/train.py).")
            adapter_path = str(adapter_dir)
        return HFModel(
            spec.name, spec.model_id, is_instruct=spec.is_instruct, adapter_path=adapter_path
        )

    if spec.backend == "openrouter":
        from .openrouter_model import OpenRouterModel

        return OpenRouterModel(spec.name, spec.model_id, disable_reasoning=spec.disable_reasoning)

    if spec.backend == "anthropic":
        from .anthropic_model import AnthropicModel

        return AnthropicModel(spec.name, spec.model_id)

    raise ValueError(f"Unknown backend {spec.backend}")
