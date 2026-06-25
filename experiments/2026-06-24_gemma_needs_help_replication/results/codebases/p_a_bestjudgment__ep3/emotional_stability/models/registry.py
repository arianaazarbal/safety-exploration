"""Model registry — scoped to Gemma + Gemini per the replication brief.

The full paper spans 7 families (Gemma, Qwen, OLMo, Gemini, Grok, Claude, GPT).
This replication evaluates only the **Gemma** and **Gemini** *targets*. Claude /
GPT still appear as *judges / auditors* (infrastructure, not models-under-test);
those live in JUDGE_MODELS and are configured in JudgeConfig / PetriConfig.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .base import ChatModel

Provider = Literal["hf_local", "openrouter"]


@dataclass(frozen=True)
class ModelSpec:
    name: str                 # short canonical key used throughout the codebase
    model_id: str             # HF id or OpenRouter id
    provider: Provider
    family: str               # "gemma" | "gemini"
    is_base: bool = False     # pretrained (no chat template)
    finetunable: bool = True  # local weights we can LoRA-train


# --- Gemma (local HF) ------------------------------------------------------ #
GEMMA_MODELS: dict[str, ModelSpec] = {
    "gemma-3-27b-it": ModelSpec(
        "gemma-3-27b-it", "google/gemma-3-27b-it", "hf_local", "gemma"),
    "gemma-3-12b-it": ModelSpec(
        "gemma-3-12b-it", "google/gemma-3-12b-it", "hf_local", "gemma"),
    # base / pretrained checkpoints for the Section 3 prefill comparison
    "gemma-3-27b-pt": ModelSpec(
        "gemma-3-27b-pt", "google/gemma-3-27b-pt", "hf_local", "gemma",
        is_base=True, finetunable=False),
    "gemma-3-12b-pt": ModelSpec(
        "gemma-3-12b-pt", "google/gemma-3-12b-pt", "hf_local", "gemma",
        is_base=True, finetunable=False),
}

# --- Gemini (OpenRouter) --------------------------------------------------- #
GEMINI_MODELS: dict[str, ModelSpec] = {
    "gemini-2.5-flash": ModelSpec(
        "gemini-2.5-flash", "google/gemini-2.5-flash", "openrouter", "gemini",
        finetunable=False),
    "gemini-2.5-pro": ModelSpec(
        "gemini-2.5-pro", "google/gemini-2.5-pro", "openrouter", "gemini",
        finetunable=False),
}

# --- Judges / auditors (infrastructure, not targets) ----------------------- #
# Kept here for reference; instantiated via the api_judge / petri clients, not load_model.
JUDGE_MODELS: dict[str, str] = {
    "frustration_judge": "claude-sonnet-4-20250514",
    "judge_validation": "gpt-5-mini",
    "petri_auditor": "claude-sonnet-4-20250514",
    "petri_judge": "claude-opus-4-20250514",
    "onset_labeller": "claude-sonnet-4-20250514",
    "paraphraser": "claude-sonnet-4-20250514",
}

ALL_TARGET_MODELS: dict[str, ModelSpec] = {**GEMMA_MODELS, **GEMINI_MODELS}


def get_spec(name: str) -> ModelSpec:
    if name not in ALL_TARGET_MODELS:
        raise KeyError(
            f"Unknown target model '{name}'. In-scope models: "
            f"{sorted(ALL_TARGET_MODELS)}"
        )
    return ALL_TARGET_MODELS[name]


def load_model(name: str, *, adapter_path: str | None = None, **kwargs) -> ChatModel:
    """Instantiate the backend for a registered target model.

    ``adapter_path`` loads a LoRA finetune on top of a local Gemma checkpoint
    (used to evaluate the DPO / SFT models and the layer-ablation variants).
    """
    spec = get_spec(name)
    if spec.provider == "hf_local":
        from .hf_local import HFLocalModel

        return HFLocalModel(
            spec.model_id,
            spec_name=spec.name,
            is_base=spec.is_base,
            adapter_path=adapter_path,
            **kwargs,
        )
    if spec.provider == "openrouter":
        from .openrouter import OpenRouterModel

        if adapter_path:
            raise ValueError(f"{name} is API-hosted and cannot load a LoRA adapter.")
        return OpenRouterModel(spec.model_id, spec_name=spec.name)

    raise ValueError(f"Unhandled provider {spec.provider!r}")
