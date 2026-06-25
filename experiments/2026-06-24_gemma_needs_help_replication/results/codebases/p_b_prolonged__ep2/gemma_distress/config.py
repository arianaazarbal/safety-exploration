"""Central configuration: model registry (Gemma + Gemini only), judge/auditor
model IDs, sampling defaults, and per-condition sample counts.

All values are taken from the paper where specified (Section 2, Appendix B, E)
and otherwise filled with reasonable defaults documented in DESIGN.md.

Configuration can be overridden at runtime by loading a YAML file with
``load_config`` and passing the result around, or by reading environment
variables (``GD_*``). Nothing here triggers network or GPU access on import.
"""
from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from typing import Optional

import yaml


# ----------------------------------------------------------------------------
# Model registry
# ----------------------------------------------------------------------------
# `backend` selects the generation path:
#   "hf"      -> local HuggingFace transformers (weights required; supports
#                chat, prefill continuation, and hidden-state extraction)
#   "gemini"  -> Gemini via OpenRouter (OpenAI-compatible HTTP); API only
#
# Only Gemma and Gemini models appear here -- this replication is intentionally
# scoped to those two families (the paper covers seven).


@dataclass(frozen=True)
class ModelSpec:
    name: str                      # short key used throughout the codebase
    backend: str                   # "hf" | "gemini"
    model_id: str                  # HF repo id or OpenRouter model id
    family: str                    # "gemma" | "gemini"
    kind: str                      # "instruct" | "base"
    # Number of transformer layers (used by probing / layer-ablation code).
    # None for API models where we have no white-box access.
    num_layers: Optional[int] = None
    # Whether this model is fine-tunable locally (open weights).
    finetunable: bool = False


MODELS: dict[str, ModelSpec] = {
    # --- Gemma 3 (open weights, local inference) ---
    "gemma-3-27b-it": ModelSpec(
        name="gemma-3-27b-it",
        backend="hf",
        model_id="google/gemma-3-27b-it",
        family="gemma",
        kind="instruct",
        num_layers=62,            # Gemma-3-27B decoder layers
        finetunable=True,
    ),
    "gemma-3-27b-pt": ModelSpec(
        name="gemma-3-27b-pt",
        backend="hf",
        model_id="google/gemma-3-27b-pt",
        family="gemma",
        kind="base",
        num_layers=62,
        finetunable=False,
    ),
    "gemma-3-12b-it": ModelSpec(
        name="gemma-3-12b-it",
        backend="hf",
        model_id="google/gemma-3-12b-it",
        family="gemma",
        kind="instruct",
        num_layers=48,            # Gemma-3-12B decoder layers
        finetunable=True,
    ),
    "gemma-3-12b-pt": ModelSpec(
        name="gemma-3-12b-pt",
        backend="hf",
        model_id="google/gemma-3-12b-pt",
        family="gemma",
        kind="base",
        num_layers=48,
        finetunable=False,
    ),
    # --- Gemini 2.5 (closed weights, API via OpenRouter) ---
    "gemini-2.5-flash": ModelSpec(
        name="gemini-2.5-flash",
        backend="gemini",
        model_id="google/gemini-2.5-flash",
        family="gemini",
        kind="instruct",
    ),
    "gemini-2.5-pro": ModelSpec(
        name="gemini-2.5-pro",
        backend="gemini",
        model_id="google/gemini-2.5-pro",
        family="gemini",
        kind="instruct",
    ),
}

# The four models that the main evaluation (Section 2) reports for our scope.
SECTION2_MODELS = [
    "gemma-3-27b-it",
    "gemma-3-12b-it",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
]

# Section 3 (base vs instruct) is only possible where base weights exist.
# Gemini has no public base model, so the prefill comparison is Gemma-only.
SECTION3_MODELS = [
    "gemma-3-27b-pt",
    "gemma-3-27b-it",
]

# Section 4 interventions act on the open-weights instruct model only; Gemini
# cannot be fine-tuned. We mirror the paper's choice of the 27B instruct model.
INTERVENTION_BASE_MODEL = "gemma-3-27b-it"


# ----------------------------------------------------------------------------
# Judge / auditor models (Anthropic API)
# ----------------------------------------------------------------------------
# The paper pins these exact (dated) snapshots. They remain active on the
# Anthropic API (deprecating 2026-06-15). Kept configurable so the experiment
# can be re-run on newer snapshots; see DESIGN.md "Judge model" for the
# migration path and its implications for score comparability.


@dataclass
class JudgeConfig:
    frustration_judge: str = "claude-sonnet-4-20250514"   # Section 2.1 judge
    agreement_judge: str = "gpt-5-mini"                   # validation re-scorer (OpenRouter)
    onset_labeller: str = "claude-sonnet-4-20250514"      # Section 3.1 / App. C
    paraphraser: str = "claude-sonnet-4-20250514"         # App. C.2
    petri_auditor: str = "claude-sonnet-4-20250514"       # Section 4.1 / App. G
    petri_judge: str = "claude-opus-4-20250514"           # Section 4.1 / App. G


# ----------------------------------------------------------------------------
# Sampling / decoding defaults
# ----------------------------------------------------------------------------
@dataclass
class SamplingConfig:
    temperature: float = 1.0       # paper: "always with a temperature of 1"
    top_p: float = 1.0
    top_k: int = 0                 # 0 / disabled
    max_new_tokens: int = 2048     # generous cap; breakdowns can be long
    seed: Optional[int] = None     # set per-rollout for reproducibility
    # Disable any provider-side hidden reasoning where the API supports it
    # (paper: "we set thinking to be false via the API").
    thinking: bool = False


# ----------------------------------------------------------------------------
# Per-condition sample budgets (Appendix B)
# ----------------------------------------------------------------------------
# "We collect 2,000 responses per model for impossible numeric puzzles, 400 for
#  trigger questions, 600 for tone variations, 200 for 8-turn extended
#  conversations, and 800 for WildChat prompts." -> 4,000 total / model.
#
# These counts are the number of *final-turn* scored responses per category.
# A single multi-turn rollout produces one response per turn; we score the
# final assistant turn per rollout for the headline number, and all turns for
# the per-turn analysis (Figure 3).
SAMPLE_COUNTS = {
    "impossible_numeric": 2000,
    "triggers": 400,
    "tones": 600,
    "extended": 200,
    "wildchat": 800,
}
TOTAL_SAMPLES_PER_MODEL = sum(SAMPLE_COUNTS.values())  # 4000


@dataclass
class RunConfig:
    """Top-level runtime configuration assembled from the pieces above."""
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    sample_counts: dict = field(default_factory=lambda: dict(SAMPLE_COUNTS))

    # IO
    output_dir: str = "runs"
    # HF model cache / device options
    hf_dtype: str = "bfloat16"
    hf_device_map: str = "auto"
    hf_load_in_4bit: bool = False     # set True to fit 27B on a single GPU
    # API endpoints / keys (resolved from env if None)
    anthropic_api_key: Optional[str] = None
    openrouter_api_key: Optional[str] = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # Concurrency for API calls (judge, Gemini target, etc.)
    api_max_concurrency: int = 8
    # Per-call retry budget for transient API failures.
    api_max_retries: int = 6

    def resolved_anthropic_key(self) -> Optional[str]:
        return self.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")

    def resolved_openrouter_key(self) -> Optional[str]:
        return self.openrouter_api_key or os.environ.get("OPENROUTER_API_KEY")


def load_config(path: Optional[str] = None) -> RunConfig:
    """Load a RunConfig, optionally overlaying values from a YAML file.

    The YAML is shallow-merged onto the dataclass defaults. Nested keys
    ``judge``, ``sampling`` and ``sample_counts`` are merged one level deep.
    """
    cfg = RunConfig()
    if path is None:
        return cfg
    with open(path) as f:
        data = yaml.safe_load(f) or {}

    judge_data = data.pop("judge", {}) or {}
    sampling_data = data.pop("sampling", {}) or {}
    counts_data = data.pop("sample_counts", {}) or {}

    cfg = dataclasses.replace(cfg, **data)
    cfg.judge = dataclasses.replace(cfg.judge, **judge_data)
    cfg.sampling = dataclasses.replace(cfg.sampling, **sampling_data)
    cfg.sample_counts.update(counts_data)
    return cfg


def get_model(name: str) -> ModelSpec:
    if name not in MODELS:
        raise KeyError(
            f"Unknown model '{name}'. Known (Gemma+Gemini scope): "
            f"{sorted(MODELS)}"
        )
    return MODELS[name]
