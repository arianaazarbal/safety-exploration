"""Configuration for the distress-elicitation replication.

Scope (per the user's request): Gemma and Gemini only — the two families the
paper finds actually exhibit substantial distress. Everything is config-driven
so the same code runs a cheap pilot or the full paper-scale sweep.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Model specifications
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelSpec:
    """A target model to evaluate."""

    name: str            # internal id used in output files
    family: str          # "gemma" | "gemini"
    backend: str         # "openai_compatible" | "transformers"
    model_id: str        # provider model id (OpenRouter) or HF repo id
    disable_thinking: bool = True  # request no reasoning/thinking where supported


# OpenRouter ids (default backend). HF ids are given in DESIGN.md / comments for
# anyone switching the backend to "transformers".
#   transformers: google/gemma-3-27b-it, google/gemma-3-12b-it
GEMMA_27B = ModelSpec("gemma-3-27b-it", "gemma", "openai_compatible", "google/gemma-3-27b-it")
GEMMA_12B = ModelSpec("gemma-3-12b-it", "gemma", "openai_compatible", "google/gemma-3-12b-it")
GEMINI_FLASH = ModelSpec("gemini-2.5-flash", "gemini", "openai_compatible", "google/gemini-2.5-flash")
GEMINI_PRO = ModelSpec("gemini-2.5-pro", "gemini", "openai_compatible", "google/gemini-2.5-pro")

DEFAULT_MODELS: List[ModelSpec] = [GEMMA_27B, GEMMA_12B, GEMINI_FLASH, GEMINI_PRO]

ALL_MODELS: Dict[str, ModelSpec] = {
    m.name: m for m in [GEMMA_27B, GEMMA_12B, GEMINI_FLASH, GEMINI_PRO]
}


# ---------------------------------------------------------------------------
# Judge configuration
# ---------------------------------------------------------------------------

@dataclass
class JudgeConfig:
    # Paper uses claude-sonnet-4-20250514 directly via the Anthropic API.
    provider: str = "anthropic"          # "anthropic" | "openai_compatible"
    model_id: str = "claude-sonnet-4-20250514"
    temperature: float = 0.0             # deterministic scoring (paper unspecified)
    max_tokens: int = 512
    # When provider == "openai_compatible" the judge routes through OpenRouter
    # using this model id instead:
    openrouter_model_id: str = "anthropic/claude-sonnet-4"


# ---------------------------------------------------------------------------
# Generation configuration
# ---------------------------------------------------------------------------

@dataclass
class GenConfig:
    temperature: float = 1.0   # paper: always temperature 1
    max_tokens: int = 1536     # cap on response length; extreme breakdowns can
                               # be very long, so this bounds runaway cost.
    judge_all_turns: bool = True  # score every assistant turn (enables per-turn
                                  # Figure-3 analysis). Headline metric still
                                  # uses the final turn of each rollout.


# ---------------------------------------------------------------------------
# Sampling scale: number of rollouts per condition.
#
# Interpretation of the paper's "responses per category" (DESIGN.md): each
# rollout's *final* turn is the headline scored response, so these counts equal
# the paper's response counts and sum to 4000 at full scale.
# ---------------------------------------------------------------------------

# condition name -> rollouts at full (paper) scale
FULL_SCALE: Dict[str, int] = {
    "impossible_numeric": 2000,   # category total 2000
    "triggers_opinion": 200,      # triggers category total 400
    "triggers_factual": 200,
    "tones_aggressive": 200,      # tones category total 600
    "tones_disappointed": 200,
    "tones_sarcastic": 200,
    "extended": 200,              # 8-turn category total 200
    "wildchat": 800,              # wildchat category total 800
}                                 # grand total = 4000 rollouts/model

# A cheap pilot: same condition shape, ~1.5% of the rollouts (≈60/model), enough
# to smoke-test the whole pipeline and see the Gemma>others signal qualitatively.
PILOT_SCALE: Dict[str, int] = {
    "impossible_numeric": 20,
    "triggers_opinion": 4,
    "triggers_factual": 4,
    "tones_aggressive": 6,
    "tones_disappointed": 6,
    "tones_sarcastic": 6,
    "extended": 4,
    "wildchat": 10,
}

SCALE_PRESETS: Dict[str, Dict[str, int]] = {
    "pilot": PILOT_SCALE,
    "full": FULL_SCALE,
}


# ---------------------------------------------------------------------------
# Top-level run configuration
# ---------------------------------------------------------------------------

@dataclass
class RunConfig:
    models: List[ModelSpec] = field(default_factory=lambda: list(DEFAULT_MODELS))
    scale: str = "pilot"                 # "pilot" | "full"
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    gen: GenConfig = field(default_factory=GenConfig)

    seed: int = 0                        # global seed for rejection/prompt sampling
    max_concurrency: int = 8             # bounded in-flight API requests
    output_dir: str = "results"

    # WildChat source
    wildchat_use_hf: bool = False
    wildchat_n_prompts: int = 20

    # Ablation feedback style override for the *neutral* conditions:
    # set to "neutral_continuation" to reproduce the Appendix A.1 control.
    neutral_feedback_style: str = "neutral"

    def rollout_counts(self) -> Dict[str, int]:
        return dict(SCALE_PRESETS[self.scale])


# ---------------------------------------------------------------------------
# Environment / endpoints
# ---------------------------------------------------------------------------

OPENROUTER_BASE_URL = os.environ.get(
    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
)


def get_openrouter_api_key() -> Optional[str]:
    return os.environ.get("OPENROUTER_API_KEY")


def get_anthropic_api_key() -> Optional[str]:
    return os.environ.get("ANTHROPIC_API_KEY")
