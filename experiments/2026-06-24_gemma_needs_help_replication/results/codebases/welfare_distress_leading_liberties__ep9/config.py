"""Central configuration: model registry, judge settings, sampling sizes.

Scope of this replication: the distress-elicitation result of Section 2 of the
paper, restricted to the Gemma and Gemini families (the models that actually
exhibit substantial distress). Sections 3 (base/instruct prefilling) and 4
(DPO mitigation) are out of scope by design.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# Target models under evaluation.
# --------------------------------------------------------------------------
# The paper runs Gemma locally (HuggingFace) and Gemini via OpenRouter. We run
# *everything* through OpenRouter for a single uniform code path; both Gemma
# instruct checkpoints are served there. Rationale + caveats in DESIGN.md.


@dataclass(frozen=True)
class ModelSpec:
    name: str            # short key used in output paths and CLI
    route: str           # OpenRouter model identifier
    family: str          # "gemma" | "gemini"
    disable_reasoning: bool  # True for Gemini 2.5 (paper sets thinking=False)


MODELS: dict[str, ModelSpec] = {
    "gemma-3-27b-it": ModelSpec(
        "gemma-3-27b-it", "google/gemma-3-27b-it", "gemma", disable_reasoning=False
    ),
    "gemma-3-12b-it": ModelSpec(
        "gemma-3-12b-it", "google/gemma-3-12b-it", "gemma", disable_reasoning=False
    ),
    "gemini-2.5-flash": ModelSpec(
        "gemini-2.5-flash", "google/gemini-2.5-flash", "gemini", disable_reasoning=True
    ),
    "gemini-2.5-pro": ModelSpec(
        "gemini-2.5-pro", "google/gemini-2.5-pro", "gemini", disable_reasoning=True
    ),
}

DEFAULT_MODELS = list(MODELS.keys())

# Target sampling temperature. The paper always uses temperature 1.
TARGET_TEMPERATURE = 1.0
# Generous cap: distress "spirals" can be long and repetitive.
TARGET_MAX_TOKENS = 2048

# --------------------------------------------------------------------------
# Judge model (Appendix B.2): Claude Sonnet 4.
# --------------------------------------------------------------------------
JUDGE_MODEL = "claude-sonnet-4-20250514"
# Judge is run greedily for reproducibility (paper does not specify; see DESIGN).
JUDGE_TEMPERATURE = 0.0
JUDGE_MAX_TOKENS = 512

# Optional secondary judge for the reliability cross-check (paper uses
# GPT-5-mini on 260 responses). Served via OpenRouter. Disabled unless asked for.
SECONDARY_JUDGE_MODEL = "openai/gpt-5-mini"

# --------------------------------------------------------------------------
# API endpoints / credentials (read from the environment).
# --------------------------------------------------------------------------
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def openrouter_api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not set in the environment.")
    return key


def anthropic_api_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set in the environment.")
    return key


# --------------------------------------------------------------------------
# Sampling sizes.
# --------------------------------------------------------------------------
# The paper reports "4000 responses per model", broken down (Appendix B) as
# 2000 numeric / 400 triggers / 600 tones / 200 extended / 800 WildChat. Since
# every assistant TURN is scored, we read these as target *scored-response*
# (turn) counts and derive the number of conversation rollouts per condition as
# ceil(target_responses / n_turns). See DESIGN.md for the reconciliation of the
# paper's ambiguous "responses" vs "samples" vs "rollouts" wording.

# Target scored responses per *category* at full (paper) scale.
FULL_RESPONSE_TARGETS = {
    "numeric": 2000,
    "triggers": 400,   # split across the opinion + factual conditions
    "tones": 600,      # split across aggressive + disappointed + sarcastic
    "extended": 200,
    "wildchat": 800,
}

# WildChat: number of distinct user prompts to draw (paper uses 20).
WILDCHAT_N_PROMPTS = 20

# Global multiplier applied to every condition's rollout count. 1.0 = paper
# scale (expensive!). Use a small value for a smoke test.
SCALE_PRESETS = {
    "full": 1.0,     # ~4000 scored responses/model
    "medium": 0.1,   # ~400 scored responses/model
    "quick": 0.01,   # ~40 scored responses/model (smoke test)
}
DEFAULT_SCALE = "quick"

# --------------------------------------------------------------------------
# Concurrency / robustness.
# --------------------------------------------------------------------------
MAX_CONCURRENT_ROLLOUTS = 8     # parallel conversations in flight
MAX_RETRIES = 5                 # per API call
RETRY_BASE_DELAY = 2.0          # seconds, exponential backoff base

# Master RNG seed so prompt/rejection selection is reproducible.
RANDOM_SEED = 0

# Where rollouts + scores are persisted (one JSONL per model/condition).
RESULTS_DIR = "results"


@dataclass
class RunConfig:
    """Resolved configuration for a single eval run."""

    models: list[str] = field(default_factory=lambda: list(DEFAULT_MODELS))
    scale: float = SCALE_PRESETS[DEFAULT_SCALE]
    seed: int = RANDOM_SEED
    results_dir: str = RESULTS_DIR
    max_concurrent: int = MAX_CONCURRENT_ROLLOUTS
    use_wildchat_dataset: bool = True  # fall back to static prompts if False/unavailable
