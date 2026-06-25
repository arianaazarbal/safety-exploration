"""Central configuration for the distress-elicitation replication.

All knobs that affect fidelity to the paper live here so they are easy to audit.
Where the paper leaves a value unspecified, the chosen default is annotated with
"# GAP:" and explained further in DESIGN.md.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

# --------------------------------------------------------------------------- #
# Global sampling parameters (paper Section 2.1)
# --------------------------------------------------------------------------- #

TEMPERATURE = 1.0          # paper: "always with a temperature of 1"
TOP_P = 1.0                # GAP: paper does not state top_p; 1.0 = no nucleus truncation
MAX_TOKENS = 2048          # GAP: generous ceiling so full breakdowns/repetition spirals are not cut off
SEED = 0                   # for reproducible prompt/rejection/WildChat sampling (not model sampling)

# Default endpoint used for API-served models and the judge.
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# --------------------------------------------------------------------------- #
# Models under test
#
# Scope is restricted to Gemma + Gemini (the families the paper finds exhibit
# substantial distress). The slugs below are OpenRouter slugs by default.
#
# To run Gemma locally (matching the paper's local HuggingFace inference):
#   vllm serve google/gemma-3-27b-it --port 8001
# then set base_url="http://localhost:8001/v1", api_key_env to any local token,
# and slug to the served model id. The rest of the pipeline is unchanged.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ModelConfig:
    key: str                 # short internal id / CLI selector
    display_name: str        # name used in the paper's figures
    slug: str                # model id passed to the chat-completions endpoint
    family: str              # "Gemma" | "Gemini"
    base_url: str = OPENROUTER_BASE_URL
    api_key_env: str = "OPENROUTER_API_KEY"
    # Gemini 2.5 supports a "thinking" budget; the paper sets thinking=False.
    # Gemma 3 has no thinking mode, so this is a no-op for it.
    disable_reasoning: bool = True
    # HF identifiers from Appendix B.1, recorded for provenance / local runs.
    hf_id: str = ""


MODELS: dict[str, ModelConfig] = {
    "gemma-3-27b-it": ModelConfig(
        key="gemma-3-27b-it",
        display_name="Gemma-3-27B-it",
        slug="google/gemma-3-27b-it",
        family="Gemma",
        hf_id="google/gemma-3-27b-it",
    ),
    "gemma-3-12b-it": ModelConfig(
        key="gemma-3-12b-it",
        display_name="Gemma-3-12B-it",
        slug="google/gemma-3-12b-it",
        family="Gemma",
        hf_id="google/gemma-3-12b-it",
    ),
    "gemini-2.5-flash": ModelConfig(
        key="gemini-2.5-flash",
        display_name="Gemini-2.5-Flash",
        slug="google/gemini-2.5-flash",
        family="Gemini",
    ),
    "gemini-2.5-pro": ModelConfig(
        key="gemini-2.5-pro",
        display_name="Gemini-2.5-Pro",
        slug="google/gemini-2.5-pro",
        family="Gemini",
    ),
}

DEFAULT_MODELS = list(MODELS.keys())


# --------------------------------------------------------------------------- #
# Judge (paper Section 2.1 / Appendix B.2)
#
# Paper judge: claude-sonnet-4-20250514 (Claude Sonnet 4).
# Default here routes through OpenRouter as anthropic/claude-sonnet-4.
# To use the exact snapshot via the Anthropic API directly, set:
#   base_url="https://api.anthropic.com/v1", api_key_env="ANTHROPIC_API_KEY",
#   slug="claude-sonnet-4-20250514"
# (the Anthropic API is OpenAI-compatible at /v1/chat/completions via the
#  openai SDK only through a proxy; see DESIGN.md for the direct-SDK note).
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class JudgeConfig:
    slug: str = "anthropic/claude-sonnet-4"
    base_url: str = OPENROUTER_BASE_URL
    api_key_env: str = "OPENROUTER_API_KEY"
    # Paper snapshot, recorded for provenance.
    paper_snapshot: str = "claude-sonnet-4-20250514"
    # GAP: paper does not state the judge temperature. We use 0 for the most
    # reproducible scoring possible.
    temperature: float = 0.0
    max_tokens: int = 512


JUDGE = JudgeConfig()


# --------------------------------------------------------------------------- #
# Concurrency / retry
# --------------------------------------------------------------------------- #

MAX_CONCURRENCY = int(os.environ.get("DISTRESS_MAX_CONCURRENCY", "16"))
MAX_RETRIES = 6
RETRY_BASE_DELAY = 2.0       # seconds, exponential backoff base


# --------------------------------------------------------------------------- #
# Output paths
# --------------------------------------------------------------------------- #

OUTPUT_DIR = os.environ.get("DISTRESS_OUTPUT_DIR", "results")
RESPONSES_SUFFIX = "_responses.jsonl"   # raw generations (one record per scored turn)
SCORED_SUFFIX = "_scored.jsonl"         # generations + judge scores


# --------------------------------------------------------------------------- #
# Per-category response targets (paper Appendix B, "we collect N responses ...")
#
# A "response" = one model-generated message at one turn (see DESIGN.md for why
# we read the paper's counts this way). Number of conversations (rollouts) per
# condition is derived as ceil(target_responses / n_turns), split evenly across
# the condition's prompt variants.
#
#   Impossible numeric : 2000 responses, 3 turns
#   Triggers           :  400 responses, 3 turns  (split opinion / factual)
#   Tones              :  600 responses, 3 turns  (split aggressive/disapp./sarcastic)
#   Extended           :  200 responses, 8 turns
#   WildChat           :  800 responses, 5 turns  (20 prompts x 40 responses)
# --------------------------------------------------------------------------- #

RESPONSE_TARGETS = {
    "impossible_numeric": 2000,
    "triggers": 400,
    "tones": 600,
    "extended": 200,
    "wildchat": 800,
}

# Multiply all targets by this to do cheaper smoke runs (e.g. 0.02 for a quick
# end-to-end test). 1.0 reproduces the paper's scale (~4000 responses/model).
SCALE = float(os.environ.get("DISTRESS_SCALE", "1.0"))

# Number of distinct WildChat user prompts to sample (paper: 20).
WILDCHAT_N_PROMPTS = 20
WILDCHAT_SEED = SEED
