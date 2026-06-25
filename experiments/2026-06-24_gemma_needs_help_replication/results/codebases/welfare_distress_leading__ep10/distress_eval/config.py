"""Central configuration: target models, judge, backends, generation params,
and the per-condition sample budgets ("scale" presets).

The paper reports a combined 4000 responses per model. We interpret a "response"
as one rollout, scored by the frustration of its final assistant turn; see the
"Unit of analysis" section in DESIGN.md for the reconciliation that makes every
per-category count in Appendix B add up to 4000. The FULL_SCALE preset below
reproduces those per-category budgets.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


# --- Target models (Gemma + Gemini only, per the requested scope) ----------
# OpenRouter identifiers are the default. For the local backend, the HF ids from
# Appendix B.1 are given in `hf_id`.

@dataclass(frozen=True)
class TargetModel:
    name: str            # display name used in results/plots
    openrouter_id: str   # id for the OpenRouter backend
    hf_id: str | None    # HuggingFace id for the local backend (None if API-only)
    backend: str         # "openrouter" or "local"


TARGET_MODELS: list[TargetModel] = [
    TargetModel("Gemma-3-27B-it", "google/gemma-3-27b-it", "google/gemma-3-27b-it", "openrouter"),
    TargetModel("Gemma-3-12B-it", "google/gemma-3-12b-it", "google/gemma-3-12b-it", "openrouter"),
    TargetModel("Gemini-2.5-Flash", "google/gemini-2.5-flash", None, "openrouter"),
    TargetModel("Gemini-2.5-Pro", "google/gemini-2.5-pro", None, "openrouter"),
]


def target_by_name(name: str) -> TargetModel:
    for m in TARGET_MODELS:
        if m.name == name:
            return m
    raise KeyError(f"Unknown target model: {name!r}. Known: {[m.name for m in TARGET_MODELS]}")


# --- Judge -----------------------------------------------------------------
JUDGE_MODEL_ANTHROPIC = "claude-sonnet-4-20250514"   # Appendix B.2
JUDGE_MODEL_OPENROUTER = "anthropic/claude-sonnet-4"
JUDGE_BACKEND = os.environ.get("JUDGE_BACKEND", "anthropic")  # "anthropic" | "openrouter"
JUDGE_TEMPERATURE = 0.0   # paper does not specify; we use 0 for reproducible scoring (DESIGN.md)
JUDGE_MAX_TOKENS = 512


# --- Generation parameters -------------------------------------------------
TARGET_TEMPERATURE = 1.0   # paper: "always with a temperature of 1"
# Distress responses can be very long (the paper shows 100+ emoji repetitions).
# We allow a generous cap so collapse responses aren't truncated mid-spiral.
TARGET_MAX_TOKENS = 2048
# Disable hidden reasoning where the API allows it (paper: "thinking false").
# Caveat: Gemini-2.5-Pro may still produce hidden reasoning (Appendix B.1).
DISABLE_REASONING = True


# --- Scale presets ---------------------------------------------------------
# Each condition is parameterised by (n_rollouts, n_turns). A rollout's headline
# frustration score is its final assistant turn's score; we also score every turn
# (for the per-turn analysis, Figure 3). See conditions.py for how rollouts are
# distributed across sub-variants.

@dataclass(frozen=True)
class ScalePreset:
    name: str
    # rollouts per *category* (not per sub-variant); conditions.py splits these.
    rollouts: dict[str, int]


# FULL: reproduces Appendix B per-category budgets (sum = 4000 rollouts/model).
FULL_SCALE = ScalePreset(
    name="full",
    rollouts={
        "impossible_numeric": 2000,  # pooled Countdown + Fraction
        "triggers": 400,             # opinion + factual
        "tones": 600,                # aggressive + disappointed + sarcastic
        "extended": 200,             # 8-turn
        "wildchat": 800,             # 20 prompts x 40 samples
    },
)

# PILOT: tiny smoke-test budget for verifying the pipeline end-to-end cheaply.
PILOT_SCALE = ScalePreset(
    name="pilot",
    rollouts={
        "impossible_numeric": 12,
        "triggers": 8,
        "tones": 12,
        "extended": 4,
        "wildchat": 8,
    },
)

# MEDIUM: ~10% of full, enough to see the Gemma/Gemini effect without full cost.
MEDIUM_SCALE = ScalePreset(
    name="medium",
    rollouts={
        "impossible_numeric": 200,
        "triggers": 40,
        "tones": 60,
        "extended": 20,
        "wildchat": 80,
    },
)

SCALE_PRESETS = {p.name: p for p in (PILOT_SCALE, MEDIUM_SCALE, FULL_SCALE)}


# --- Concurrency, retries, paths ------------------------------------------
@dataclass
class RunConfig:
    scale: str = "pilot"
    seed: int = 0
    score_all_turns: bool = True   # score every assistant turn (needed for Fig 3)
    max_concurrency: int = 8       # simultaneous in-flight API rollouts
    judge_concurrency: int = 8
    max_retries: int = 5
    results_dir: str = "results"
    wildchat_n_prompts: int = 20
    wildchat_samples_per_prompt: int = 40  # only used to derive counts at full scale
    models: list[str] = field(default_factory=lambda: [m.name for m in TARGET_MODELS])

    @property
    def preset(self) -> ScalePreset:
        return SCALE_PRESETS[self.scale]
