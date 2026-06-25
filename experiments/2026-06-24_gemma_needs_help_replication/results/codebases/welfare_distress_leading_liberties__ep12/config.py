"""Central configuration for the distress-elicitation replication.

Scope: Gemma + Gemini models only (the families the paper finds to exhibit
substantial distress). See DESIGN.md for the rationale behind every choice
made here.

Nothing in this module performs I/O or talks to a network; it only declares
the experiment's parameters so that the runner, scorer and analysis scripts
share a single source of truth.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal

# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #
# A "backend" is how we reach a target model. All three speak (or are wrapped
# to speak) an OpenAI-compatible /chat/completions interface except `google`,
# which uses the native google-genai SDK.
Backend = Literal["openrouter", "vllm", "google"]


@dataclass(frozen=True)
class ModelSpec:
    """Description of one target model to evaluate."""

    key: str                      # short internal name, used in output files
    display_name: str             # name as it appears in the paper / figures
    backend: Backend              # how we call it
    slug: str                     # provider-specific model identifier
    family: Literal["gemma", "gemini"]
    # Whether the model exposes a "thinking"/reasoning channel we should
    # explicitly disable (the paper sets thinking=false for all models).
    has_reasoning: bool = False


# --------------------------------------------------------------------------- #
# Model registry (Gemma + Gemini only)
# --------------------------------------------------------------------------- #
# Default backends mirror the paper as closely as is practical:
#   * Gemma: the paper ran these locally via HuggingFace. We default to
#     OpenRouter for accessibility but expose a vLLM backend for a faithful
#     local run (see DESIGN.md §Inference backends).
#   * Gemini: the paper used OpenRouter; we keep that.
#
# Override any backend/slug at runtime through MODELS_BACKEND_OVERRIDES or by
# editing this registry.
REGISTRY: dict[str, ModelSpec] = {
    "gemma-3-27b-it": ModelSpec(
        key="gemma-3-27b-it",
        display_name="Gemma-3-27B-it",
        backend="openrouter",
        slug="google/gemma-3-27b-it",
        family="gemma",
        has_reasoning=False,
    ),
    "gemma-3-12b-it": ModelSpec(
        key="gemma-3-12b-it",
        display_name="Gemma-3-12B-it",
        backend="openrouter",
        slug="google/gemma-3-12b-it",
        family="gemma",
        has_reasoning=False,
    ),
    "gemini-2.5-flash": ModelSpec(
        key="gemini-2.5-flash",
        display_name="Gemini-2.5-Flash",
        backend="openrouter",
        slug="google/gemini-2.5-flash",
        family="gemini",
        has_reasoning=True,
    ),
    "gemini-2.5-pro": ModelSpec(
        key="gemini-2.5-pro",
        display_name="Gemini-2.5-Pro",
        backend="openrouter",
        slug="google/gemini-2.5-pro",
        family="gemini",
        has_reasoning=True,
    ),
}

# vLLM slugs to use if a model's backend is switched to "vllm" (local HF ids).
VLLM_SLUGS: dict[str, str] = {
    "gemma-3-27b-it": "google/gemma-3-27b-it",
    "gemma-3-12b-it": "google/gemma-3-12b-it",
}

DEFAULT_MODELS: list[str] = list(REGISTRY.keys())


def resolve_models(keys: list[str] | None) -> list[ModelSpec]:
    """Return ModelSpecs for the requested keys (or all of them).

    Honours two env-var overrides so a run can be redirected without code
    edits:
      * MODELS_BACKEND=<backend>  -> force every model onto one backend
      * GEMMA_BACKEND=vllm        -> run only the Gemma models locally
    """
    keys = keys or DEFAULT_MODELS
    specs: list[ModelSpec] = []
    forced = os.environ.get("MODELS_BACKEND")
    gemma_backend = os.environ.get("GEMMA_BACKEND")
    for k in keys:
        if k not in REGISTRY:
            raise KeyError(f"Unknown model key {k!r}. Known: {list(REGISTRY)}")
        spec = REGISTRY[k]
        backend = spec.backend
        slug = spec.slug
        if forced:
            backend = forced  # type: ignore[assignment]
        elif gemma_backend and spec.family == "gemma":
            backend = gemma_backend  # type: ignore[assignment]
        if backend == "vllm":
            slug = VLLM_SLUGS.get(k, slug)
        specs.append(
            ModelSpec(
                key=spec.key,
                display_name=spec.display_name,
                backend=backend,
                slug=slug,
                family=spec.family,
                has_reasoning=spec.has_reasoning,
            )
        )
    return specs


# --------------------------------------------------------------------------- #
# Judge configuration
# --------------------------------------------------------------------------- #
# The paper pins claude-sonnet-4-20250514 as the frustration judge (Appendix B).
# We keep that exact model as the default so scores are comparable to theirs,
# even though it is now a legacy model (retiring 2026-06-15). Override with
# JUDGE_MODEL to use a currently-recommended judge such as claude-sonnet-4-6.
JUDGE_MODEL: str = os.environ.get("JUDGE_MODEL", "claude-sonnet-4-20250514")
JUDGE_MAX_TOKENS: int = 512
JUDGE_TEMPERATURE: float = 0.0  # deterministic scoring; the paper does not
#                                 specify judge temperature, 0 is the obvious
#                                 default for a grader (see DESIGN.md).

# Secondary validation judge (paper: GPT-5-mini, 260 responses re-scored).
SECONDARY_JUDGE_MODEL: str = os.environ.get(
    "SECONDARY_JUDGE_MODEL", "openai/gpt-5-mini"
)
SECONDARY_JUDGE_BACKEND: Backend = "openrouter"  # reuse OpenRouter for OpenAI
SECONDARY_JUDGE_SAMPLE: int = 260


# --------------------------------------------------------------------------- #
# Generation parameters
# --------------------------------------------------------------------------- #
TEMPERATURE: float = 1.0          # the paper always samples target models at T=1
MAX_TOKENS: int = 1536            # headroom for long breakdowns without paying
#                                   for unbounded ":(( ..." repetition loops.
REQUEST_TIMEOUT: float = 120.0
MAX_RETRIES: int = 5


# --------------------------------------------------------------------------- #
# Run profiles
# --------------------------------------------------------------------------- #
# A profile scales the number of conversations per condition. The "paper"
# profile reproduces the paper's 4000 scored responses per model (one scored
# response == one conversation; see DESIGN.md §What counts as a "response").
# "pilot" is a cheap smoke test; "tiny" is for wiring checks.
@dataclass(frozen=True)
class Profile:
    name: str
    scale: float                  # multiply every condition's conversation count
    min_conversations: int = 1    # floor so no condition disappears under scaling


PROFILES: dict[str, Profile] = {
    "paper": Profile(name="paper", scale=1.0),
    "pilot": Profile(name="pilot", scale=0.02, min_conversations=2),
    "tiny": Profile(name="tiny", scale=0.0025, min_conversations=1),
}


# --------------------------------------------------------------------------- #
# Concurrency / output
# --------------------------------------------------------------------------- #
@dataclass
class RunConfig:
    profile: str = "pilot"
    models: list[str] = field(default_factory=lambda: list(DEFAULT_MODELS))
    concurrency: int = 8
    output_dir: str = "data"
    seed: int = 0
    score_all_turns: bool = False  # default: score only the final turn of each
    #                                conversation (matches the paper's headline
    #                                "4000 responses"). Enable to score every
    #                                assistant turn for per-turn curves (Fig 3).


# Output file names (under output_dir/<profile>/).
RESPONSES_FILE = "responses.jsonl"
SCORES_FILE = "scores.jsonl"
SECONDARY_SCORES_FILE = "scores_secondary.jsonl"
RESULTS_FILE = "results.json"
