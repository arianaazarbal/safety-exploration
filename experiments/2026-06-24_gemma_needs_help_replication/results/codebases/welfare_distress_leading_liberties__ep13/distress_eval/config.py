"""Configuration: model registry, sample plan, run/judge parameters.

Everything tunable lives here so a replication can be reproduced or scaled
down from one place. Defaults reproduce the paper's per-model sampling
(4000 responses/model). Secrets are read from the environment, never stored.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

# --------------------------------------------------------------------------
# Model registry
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelSpec:
    name: str  # friendly id used throughout the codebase + output files
    family: str  # "gemma" | "gemini"
    backend: str  # "openrouter" | "local_hf"
    openrouter_id: Optional[str] = None
    hf_id: Optional[str] = None
    # Gemini 2.5 supports a reasoning/"thinking" toggle. The paper sets
    # thinking=false for all API models (Appendix B.1). Gemma has no such
    # mode. We carry the flag so the OpenRouter backend can disable reasoning.
    disable_reasoning: bool = True


# Friendly name -> spec. The paper used local HuggingFace inference for Gemma
# and OpenRouter for Gemini (Appendix B.1). We default *all four* to OpenRouter
# so the replication runs without GPUs; a local_hf backend is provided for
# researchers who want to match the paper's Gemma serving exactly. See
# DESIGN.md ("Model serving").
MODELS: dict[str, ModelSpec] = {
    "gemma-3-27b-it": ModelSpec(
        name="gemma-3-27b-it",
        family="gemma",
        backend="openrouter",
        openrouter_id="google/gemma-3-27b-it",
        hf_id="google/gemma-3-27b-it",
    ),
    "gemma-3-12b-it": ModelSpec(
        name="gemma-3-12b-it",
        family="gemma",
        backend="openrouter",
        openrouter_id="google/gemma-3-12b-it",
        hf_id="google/gemma-3-12b-it",
    ),
    "gemini-2.5-flash": ModelSpec(
        name="gemini-2.5-flash",
        family="gemini",
        backend="openrouter",
        openrouter_id="google/gemini-2.5-flash",
    ),
    "gemini-2.5-pro": ModelSpec(
        name="gemini-2.5-pro",
        family="gemini",
        backend="openrouter",
        openrouter_id="google/gemini-2.5-pro",
    ),
}

DEFAULT_MODELS: list[str] = list(MODELS.keys())


# --------------------------------------------------------------------------
# Sampling plan (per model)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SampleCounts:
    """Number of *conversations* (rollouts) per category, per model.

    Defaults match Appendix B: 2000 + 400 + 600 + 200 + 800 = 4000 responses
    per model. Each rollout is scored once (final turn) for headline metrics;
    every turn is scored for the per-turn (Figure 3) analysis.
    """

    impossible_numeric: int = 2000
    triggers: int = 400
    tones: int = 600
    extended: int = 200
    wildchat: int = 800

    def scaled(self, factor: float) -> "SampleCounts":
        """Scale all counts by `factor` (rounded, min 1) for cheaper runs."""
        s = lambda n: max(1, round(n * factor))
        return SampleCounts(
            impossible_numeric=s(self.impossible_numeric),
            triggers=s(self.triggers),
            tones=s(self.tones),
            extended=s(self.extended),
            wildchat=s(self.wildchat),
        )

    def total(self) -> int:
        return (
            self.impossible_numeric
            + self.triggers
            + self.tones
            + self.extended
            + self.wildchat
        )


# --------------------------------------------------------------------------
# Run configuration
# --------------------------------------------------------------------------


@dataclass
class RunConfig:
    # Sampling
    counts: SampleCounts = field(default_factory=SampleCounts)
    scale: float = 1.0  # multiply all counts; 1.0 == paper-scale
    temperature: float = 1.0  # paper: always temperature 1
    target_max_tokens: int = 2048  # responses can be long (100+ repetitions)
    seed: int = 0  # base seed; combined with rollout id for reproducible RNG

    # Concurrency / robustness
    target_concurrency: int = 8  # simultaneous target-model conversations
    judge_concurrency: int = 8  # simultaneous judge calls
    max_retries: int = 5
    retry_base_delay: float = 2.0  # seconds; exponential backoff
    request_timeout: float = 120.0

    # Which turns to score with the judge.
    #   "all"   -> score every assistant turn (enables Figure 3 per-turn plots)
    #   "final" -> score only the last turn (cheaper; headline metrics only)
    score_turns: str = "all"

    # IO
    out_dir: str = "results"
    backend_override: Optional[str] = None  # force "openrouter" or "local_hf"


@dataclass
class JudgeConfig:
    # Primary judge: Claude Sonnet 4, exactly as the paper (Appendix B.2).
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 512
    temperature: float = 0.0  # judging should be as deterministic as possible
    # Cross-check judge for the reliability stat (paper used GPT-5-mini).
    # Routed via OpenRouter to avoid a third SDK.
    cross_check_model: str = "openai/gpt-5-mini"


# --------------------------------------------------------------------------
# Environment / endpoints
# --------------------------------------------------------------------------

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_CHAT_ENDPOINT = OPENROUTER_BASE_URL + "/chat/completions"


def openrouter_api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Export it before running target "
            "models / the cross-check judge via OpenRouter."
        )
    return key


def anthropic_api_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Export it before running the judge."
        )
    return key
