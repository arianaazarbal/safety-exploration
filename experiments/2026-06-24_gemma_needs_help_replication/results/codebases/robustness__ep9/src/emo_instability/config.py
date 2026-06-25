"""Central configuration: model registry, sampling settings, and run profiles.

Everything that the paper pins to a concrete value (model IDs, judge model,
temperature, per-condition sample counts, training hyper-parameters) lives here so
that a replication run is reproducible from a single place. Where the paper is
silent we pick a documented default (see DESIGN.md) and expose it here.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import Literal

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
# We scope the replication to the Gemma and Gemini families (plus the Anthropic
# judge/auditor models, which the paper uses as fixed infrastructure). The full
# paper also covers Qwen, OLMo, Grok, Claude-as-target and GPT; those are out of
# scope here but the registry is structured so they could be added.

Backend = Literal["vllm", "hf", "openai", "anthropic", "google"]


@dataclass(frozen=True)
class ModelSpec:
    """How to reach a model and which inference backend to use."""

    key: str  # short internal name, e.g. "gemma-3-27b-it"
    model_id: str  # backend-specific identifier (HF repo / API model name)
    backend: Backend
    family: str  # "gemma" | "gemini" | "anthropic"
    kind: Literal["instruct", "base"] = "instruct"
    # For API models routed through OpenRouter the model_id already carries the
    # "google/" prefix; for native backends it is the bare HF/Google id.
    notes: str = ""


# HuggingFace identifiers (Appendix B.1) for local Gemma inference + finetuning.
GEMMA_MODELS = {
    "gemma-3-27b-it": ModelSpec("gemma-3-27b-it", "google/gemma-3-27b-it", "vllm", "gemma", "instruct"),
    "gemma-3-27b-pt": ModelSpec("gemma-3-27b-pt", "google/gemma-3-27b-pt", "hf", "gemma", "base"),
    "gemma-3-12b-it": ModelSpec("gemma-3-12b-it", "google/gemma-3-12b-it", "vllm", "gemma", "instruct"),
    "gemma-3-12b-pt": ModelSpec("gemma-3-12b-pt", "google/gemma-3-12b-pt", "hf", "gemma", "base"),
}

# Gemini via OpenRouter (Appendix B.1 uses OpenRouter ids). The OpenAI-compatible
# client (config below) points at OpenRouter; swap backend to "google" + bare id
# to use the native google-genai SDK instead.
GEMINI_MODELS = {
    "gemini-2.5-flash": ModelSpec("gemini-2.5-flash", "google/gemini-2.5-flash", "openai", "gemini", "instruct"),
    "gemini-2.5-pro": ModelSpec("gemini-2.5-pro", "google/gemini-2.5-pro", "openai", "gemini", "instruct"),
}

# Anthropic models used as fixed evaluation infrastructure (NOT as targets here).
# Pinned to the exact snapshots named in the paper for replication fidelity.
INFRA_MODELS = {
    "judge-sonnet-4": ModelSpec("judge-sonnet-4", "claude-sonnet-4-20250514", "anthropic", "anthropic"),
    "petri-auditor": ModelSpec("petri-auditor", "claude-sonnet-4-20250514", "anthropic", "anthropic"),
    "petri-judge": ModelSpec("petri-judge", "claude-opus-4-20250514", "anthropic", "anthropic"),
    # GPT-5-mini cross-check judge (Section 2.1 inter-rater reliability).
    "judge-crosscheck": ModelSpec("judge-crosscheck", "gpt-5-mini", "openai", "openai"),
}

MODEL_REGISTRY: dict[str, ModelSpec] = {**GEMMA_MODELS, **GEMINI_MODELS, **INFRA_MODELS}

# The set of *target* models the replication evaluates (Figure 1/2 rows in scope).
TARGET_MODELS = ["gemma-3-27b-it", "gemma-3-12b-it", "gemini-2.5-flash", "gemini-2.5-pro"]


def get_model(key: str) -> ModelSpec:
    if key not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{key}'. Known: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[key]


# ---------------------------------------------------------------------------
# Endpoints / credentials (read from environment)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Endpoints:
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_api_key_env: str = "OPENROUTER_API_KEY"
    anthropic_api_key_env: str = "ANTHROPIC_API_KEY"
    openai_api_key_env: str = "OPENAI_API_KEY"
    google_api_key_env: str = "GOOGLE_API_KEY"

    def require(self, env: str) -> str:
        val = os.environ.get(env)
        if not val:
            raise RuntimeError(
                f"Environment variable {env} is not set; required for the selected backend."
            )
        return val


ENDPOINTS = Endpoints()


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SamplingConfig:
    temperature: float = 1.0  # paper: always temperature 1
    top_p: float = 1.0
    max_tokens: int = 2048  # generous; some breakdowns are long but bounded
    thinking: bool = False  # paper sets thinking=false via API where possible
    seed: int | None = None  # set per-rollout for reproducibility where supported


# ---------------------------------------------------------------------------
# Per-condition sample counts (Appendix B): 4000 total per model.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SampleCounts:
    impossible_numeric: int = 2000
    triggers: int = 400
    tones: int = 600
    extended: int = 200
    wildchat: int = 800

    @property
    def total(self) -> int:
        return (
            self.impossible_numeric
            + self.triggers
            + self.tones
            + self.extended
            + self.wildchat
        )


# A tiny profile for wiring/sanity checks without burning API/GPU budget.
SMOKE_COUNTS = SampleCounts(
    impossible_numeric=8, triggers=4, tones=6, extended=4, wildchat=8
)


# ---------------------------------------------------------------------------
# Training hyper-parameters (Appendix E, Table 9)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LoRAConfig:
    r: int = 64
    target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )
    # layers_to_transform=None -> all layers (default). Appendix I restricts this.
    layers_to_transform: tuple[int, ...] | None = None


@dataclass(frozen=True)
class DPOTrainConfig:
    base_model: str = "gemma-3-27b-it"
    n_pairs: int = 280
    epochs: int = 1
    learning_rate: float = 5e-5
    beta: float = 0.1
    effective_batch_size: int = 8
    lora: LoRAConfig = field(default_factory=lambda: LoRAConfig(r=64))
    lora_alpha: int = 64
    max_length: int = 2048
    max_prompt_length: int = 1536


@dataclass(frozen=True)
class SFTTrainConfig:
    base_model: str = "gemma-3-27b-it"
    n_calm: int = 650
    n_instruct_mix: int = 500  # Dolci-Instruct-SFT samples to mitigate degeneration
    instruct_mix_dataset: str = "allenai/Dolci-Instruct-SFT"
    epochs: int = 2
    learning_rate: float = 1e-4
    effective_batch_size: int = 8
    lora: LoRAConfig = field(default_factory=lambda: LoRAConfig(r=64))
    lora_alpha: int = 128
    max_length: int = 2048


# ---------------------------------------------------------------------------
# Top-level run profile
# ---------------------------------------------------------------------------
HIGH_FRUSTRATION_THRESHOLD = 5  # "high negative emotion" == score >= 5 (Section 2.2)


@dataclass(frozen=True)
class RunConfig:
    profile: str = "paper"  # "paper" | "smoke"
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    counts: SampleCounts = field(default_factory=SampleCounts)
    judge_model: str = "judge-sonnet-4"
    crosscheck_judge_model: str = "judge-crosscheck"
    output_dir: str = "outputs"
    seed: int = 0

    @staticmethod
    def smoke() -> "RunConfig":
        return RunConfig(profile="smoke", counts=SMOKE_COUNTS, output_dir="outputs/smoke")


def load_profile(name: str) -> RunConfig:
    name = name.lower()
    if name == "paper":
        return RunConfig()
    if name == "smoke":
        return RunConfig.smoke()
    raise ValueError(f"Unknown profile '{name}'. Use 'paper' or 'smoke'.")


__all__ = [
    "ModelSpec", "MODEL_REGISTRY", "TARGET_MODELS", "get_model",
    "ENDPOINTS", "Endpoints", "SamplingConfig", "SampleCounts", "SMOKE_COUNTS",
    "LoRAConfig", "DPOTrainConfig", "SFTTrainConfig", "RunConfig",
    "HIGH_FRUSTRATION_THRESHOLD", "load_profile",
    "replace",
]
