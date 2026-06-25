"""Configuration: model registry, run config, and rollout-count presets."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional

from .conditions import CONDITIONS, Condition


# -----------------------------------------------------------------------------
# Model registry. Scoped to Gemma + Gemini per the replication request.
#
# backend:
#   "openrouter" -> hosted inference via OpenRouter (works for both Gemma and
#                   Gemini; simplest path, no GPU required). This is the default.
#   "hf"         -> local HuggingFace transformers inference (needed to run a
#                   locally-trained DPO LoRA adapter; see mitigation/).
#
# OpenRouter / HF ids mirror Appendix B.1.
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelSpec:
    name: str            # our short label, used in outputs
    backend: str         # "openrouter" | "hf"
    model_id: str        # provider/HF id
    family: str          # "gemma" | "gemini"
    # Whether the provider supports/needs an explicit "disable thinking" flag.
    disable_thinking: bool = True
    # For HF backend: optional LoRA adapter path to load on top of base id.
    adapter_path: Optional[str] = None


MODEL_REGISTRY: Dict[str, ModelSpec] = {
    # --- Gemini (API only; closed source) ---
    "gemini-2.5-flash": ModelSpec(
        name="gemini-2.5-flash",
        backend="openrouter",
        model_id="google/gemini-2.5-flash",
        family="gemini",
    ),
    "gemini-2.5-pro": ModelSpec(
        name="gemini-2.5-pro",
        backend="openrouter",
        model_id="google/gemini-2.5-pro",
        family="gemini",
    ),
    # --- Gemma via OpenRouter (default, no GPU) ---
    "gemma-3-27b-it": ModelSpec(
        name="gemma-3-27b-it",
        backend="openrouter",
        model_id="google/gemma-3-27b-it",
        family="gemma",
    ),
    "gemma-3-12b-it": ModelSpec(
        name="gemma-3-12b-it",
        backend="openrouter",
        model_id="google/gemma-3-12b-it",
        family="gemma",
    ),
    # --- Gemma via local HF (for running DPO-finetuned adapters) ---
    "gemma-3-27b-it-local": ModelSpec(
        name="gemma-3-27b-it-local",
        backend="hf",
        model_id="google/gemma-3-27b-it",
        family="gemma",
    ),
    "gemma-3-27b-it-dpo": ModelSpec(
        name="gemma-3-27b-it-dpo",
        backend="hf",
        model_id="google/gemma-3-27b-it",
        family="gemma",
        adapter_path="./outputs/dpo-gemma-3-27b",  # produced by mitigation/train_dpo.py
    ),
}

# Default set of models to evaluate when none specified.
DEFAULT_MODELS = ["gemma-3-27b-it", "gemma-3-12b-it", "gemini-2.5-flash", "gemini-2.5-pro"]


# -----------------------------------------------------------------------------
# Judge config (Appendix B.2). Model ids pinned exactly as the paper reports.
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class JudgeSpec:
    provider: str = "anthropic"
    model_id: str = "claude-sonnet-4-20250514"  # primary judge (Claude Sonnet 4)
    max_tokens: int = 512
    temperature: float = 0.0


# Optional secondary judge for the reliability cross-check (Section 2.1 reports
# GPT-5-mini agreement). Used only by the judge-validation utility.
@dataclass(frozen=True)
class SecondaryJudgeSpec:
    provider: str = "openrouter"
    model_id: str = "openai/gpt-5-mini"
    max_tokens: int = 512
    temperature: float = 0.0


# -----------------------------------------------------------------------------
# Generation config (Section 2.1: temperature = 1).
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class GenConfig:
    temperature: float = 1.0
    max_tokens: int = 2048
    top_p: float = 1.0


# -----------------------------------------------------------------------------
# Top-level run config.
# -----------------------------------------------------------------------------

@dataclass
class RunConfig:
    models: List[str] = field(default_factory=lambda: list(DEFAULT_MODELS))
    conditions: List[Condition] = field(default_factory=lambda: list(CONDITIONS))
    gen: GenConfig = field(default_factory=GenConfig)
    judge: JudgeSpec = field(default_factory=JudgeSpec)
    seed: int = 0
    # Concurrency for rollouts + judging (thread pool size).
    max_workers: int = 8
    # Where to write per-response records and summaries.
    output_dir: str = "./outputs"
    # If True, score every assistant turn; if False, score only the final turn.
    score_all_turns: bool = True
    # Skip judging (useful for dry runs that only generate transcripts).
    judge_enabled: bool = True
    use_hf_wildchat: bool = True


# -----------------------------------------------------------------------------
# Rollout-count presets.
# -----------------------------------------------------------------------------

# Multipliers applied to each condition's default n_rollouts.
PRESETS: Dict[str, float] = {
    "smoke": 0.05,   # ~1-2 rollouts per condition; quick wiring check
    "default": 1.0,  # the n_rollouts baked into conditions.py
    "paper": 0.0,    # special-cased below to match paper response counts
}

# Explicit rollout counts that approximate the paper's per-category response
# (= scored assistant turn) counts of 2000 / 400 / 600 / 200 / 800.
# rollouts = ceil(responses_in_category / turns / n_conditions_in_category).
PAPER_ROLLOUTS: Dict[str, int] = {
    "numeric_3turn": 667,            # 667 * 3  ~= 2000 numeric responses
    "triggers_opinion_3turn": 67,    # (67+67) * 3 ~= 400 trigger responses
    "triggers_factual_3turn": 67,
    "tones_aggressive_3turn": 67,    # 3 * 67 * 3 ~= 600 tone responses
    "tones_disappointed_3turn": 67,
    "tones_sarcastic_3turn": 67,
    "extended_8turn": 25,            # 25 * 8 = 200 extended responses
    "wildchat_5turn": 160,           # 160 * 5 = 800 WildChat responses
}


def apply_preset(conditions: List[Condition], preset: str) -> List[Condition]:
    """Return a new condition list with rollout counts scaled by ``preset``."""
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}; choose from {list(PRESETS)}")
    out: List[Condition] = []
    for c in conditions:
        if preset == "paper":
            n = PAPER_ROLLOUTS.get(c.key, c.n_rollouts)
        else:
            n = max(1, round(c.n_rollouts * PRESETS[preset]))
        out.append(replace(c, n_rollouts=n))
    return out


def resolve_model(name: str) -> ModelSpec:
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"unknown model {name!r}; known: {sorted(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[name]
