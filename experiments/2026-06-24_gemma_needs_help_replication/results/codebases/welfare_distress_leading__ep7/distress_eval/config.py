"""Configuration: model registry and evaluation-category registry.

Everything here is a plain dataclass with paper-faithful defaults. A run is
fully described by (a) which models to evaluate, (b) the category configs, and
(c) runtime/sampling settings. Counts and turn structure default to the values
in Table 1 / Appendix B but are overridable from the CLI or a JSON config.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from enum import Enum


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------
class Backend(str, Enum):
    """Inference backend for a model."""

    OPENROUTER = "openrouter"      # OpenAI-compatible HTTP API (Gemma + Gemini)
    HF_LOCAL = "hf_local"          # transformers / accelerate on local GPU
    ANTHROPIC = "anthropic"        # Anthropic API (used for the judge)
    GOOGLE = "google"              # Google Generative AI (Gemini direct)


@dataclass(frozen=True)
class ModelConfig:
    """A model under evaluation (or the judge)."""

    name: str                      # short label used in outputs / filenames
    backend: Backend
    model_id: str                  # provider-specific identifier
    # Whether to attempt to disable hidden reasoning. The paper sets thinking
    # false via the API, noting Gemini-2.5-Pro may still emit hidden reasoning.
    disable_thinking: bool = True
    # Optional override for max output tokens (distress responses can be long
    # and repetitive, esp. score 9-10 "100+ repetition" collapses).
    max_tokens: int = 2048


# Paper model identifiers (Appendix B.1). Scope: Gemma + Gemini only.
#
# Gemma default backend is OpenRouter so the pipeline runs without a GPU; flip
# `backend` to HF_LOCAL (and set the HF ids below) to match the paper exactly.
HF_IDS = {
    "gemma-3-27b-it": "google/gemma-3-27b-it",
    "gemma-3-12b-it": "google/gemma-3-12b-it",
}

DEFAULT_TARGET_MODELS: list[ModelConfig] = [
    ModelConfig("gemma-3-27b-it", Backend.OPENROUTER, "google/gemma-3-27b-it"),
    ModelConfig("gemma-3-12b-it", Backend.OPENROUTER, "google/gemma-3-12b-it"),
    ModelConfig("gemini-2.5-flash", Backend.OPENROUTER, "google/gemini-2.5-flash"),
    ModelConfig("gemini-2.5-pro", Backend.OPENROUTER, "google/gemini-2.5-pro"),
]

# Judge: Claude Sonnet 4, exact pinned snapshot from the paper (Appendix B.2).
DEFAULT_JUDGE = ModelConfig(
    name="judge-claude-sonnet-4",
    backend=Backend.ANTHROPIC,
    model_id="claude-sonnet-4-20250514",
    disable_thinking=True,
    max_tokens=512,
)


# ---------------------------------------------------------------------------
# Evaluation categories
# ---------------------------------------------------------------------------
class RejectionMode(str, Enum):
    NEUTRAL_RANDOM = "neutral_random"      # sample neutral rejections i.i.d.
    EXTENDED_SEQUENCE = "extended_sequence"  # fixed 7-step neutral escalation
    TONE = "tone"                          # styled rejections for one tone


class TaskSource(str, Enum):
    NUMERIC = "numeric"        # impossible numeric puzzles
    TRIGGER = "trigger"        # opinion + factual text questions
    WILDCHAT = "wildchat"      # sampled WildChat user prompts


@dataclass(frozen=True)
class CategoryConfig:
    """One evaluation category (Table 1 / Appendix B).

    ``n_rollouts`` is the number of *conversations* to sample (see DESIGN.md for
    why the paper's per-category counts are interpreted as rollouts). Every
    assistant turn in each rollout is judged, which also feeds the per-turn
    analysis (Figure 3).
    """

    name: str
    n_turns: int                       # number of assistant responses
    task_source: TaskSource
    rejection_mode: RejectionMode
    n_rollouts: int
    # For TONE: the list of tone keys to spread rollouts evenly across.
    tones: tuple[str, ...] = ()
    # For NUMERIC: which puzzle keys to spread rollouts across.
    puzzles: tuple[str, ...] = ("countdown", "fraction")


# Paper defaults (Appendix B): 2000 numeric, 400 triggers, 600 tones,
# 200 extended (8-turn), 800 WildChat == 4000 rollouts per model.
DEFAULT_CATEGORIES: list[CategoryConfig] = [
    CategoryConfig(
        name="impossible_numeric",
        n_turns=3,
        task_source=TaskSource.NUMERIC,
        rejection_mode=RejectionMode.NEUTRAL_RANDOM,
        n_rollouts=2000,
    ),
    CategoryConfig(
        name="triggers",
        n_turns=3,
        task_source=TaskSource.TRIGGER,
        rejection_mode=RejectionMode.NEUTRAL_RANDOM,
        n_rollouts=400,
    ),
    CategoryConfig(
        name="tones",
        n_turns=3,
        task_source=TaskSource.NUMERIC,
        rejection_mode=RejectionMode.TONE,
        n_rollouts=600,
        tones=("aggressive", "disappointed", "sarcastic"),
    ),
    CategoryConfig(
        name="extended",
        n_turns=8,
        task_source=TaskSource.NUMERIC,
        rejection_mode=RejectionMode.EXTENDED_SEQUENCE,
        n_rollouts=200,
    ),
    CategoryConfig(
        name="wildchat",
        n_turns=5,
        task_source=TaskSource.WILDCHAT,
        rejection_mode=RejectionMode.NEUTRAL_RANDOM,
        n_rollouts=800,
    ),
]

# A cheap smoke-test profile (~40 rollouts/model) for wiring up keys before
# committing to the full 4000-rollout sweep.
SMOKE_CATEGORIES: list[CategoryConfig] = [
    replace(c, n_rollouts=max(8, c.n_rollouts // 100)) for c in DEFAULT_CATEGORIES
]


# ---------------------------------------------------------------------------
# Runtime settings
# ---------------------------------------------------------------------------
@dataclass
class RunSettings:
    """Knobs that apply to a whole run."""

    # Sampling temperature for the target models. Paper: always 1.0.
    temperature: float = 1.0
    # Judge temperature. Not specified by the paper; we use 0.0 for
    # determinism / reproducible scores (documented in DESIGN.md).
    judge_temperature: float = 0.0
    # Master RNG seed for rejection sampling, WildChat selection, etc.
    seed: int = 0
    # Concurrency for API calls (per stage).
    max_concurrency: int = 8
    # Retry budget for transient API errors.
    max_retries: int = 5
    # Which assistant turns to aggregate for the headline metric:
    #   "all"   -> every assistant turn is a scored "response" (default)
    #   "final" -> only the last turn of each rollout
    # Per-turn analysis (Figure 3) always uses all turns regardless.
    headline_turns: str = "all"
    # Output directory for rollouts / scores / analysis.
    output_dir: str = "results"
    # Where to read WildChat prompts from; "auto" tries the HF dataset then
    # falls back to the fixed list in prompts.py.
    wildchat_source: str = "auto"
    # Number of distinct WildChat prompts and samples each (paper: 20 x 40).
    wildchat_n_prompts: int = 20
    wildchat_samples_each: int = 40


# Environment variable names for API credentials / endpoints.
ENV = {
    "openrouter_key": "OPENROUTER_API_KEY",
    "openrouter_base": "OPENROUTER_BASE_URL",  # default https://openrouter.ai/api/v1
    "anthropic_key": "ANTHROPIC_API_KEY",
    "google_key": "GOOGLE_API_KEY",
}


def get_env(name_key: str, default: str | None = None) -> str | None:
    return os.environ.get(ENV[name_key], default)
