"""Central configuration for the emotional-instability replication.

This file defines:
  * the target models (scope: Gemma + Gemini only),
  * which backend serves each model,
  * the judge model,
  * sampling parameters (temperature, sample counts per condition).

All values that the paper specifies are taken verbatim where possible; where the
paper is silent the choice is documented in DESIGN.md.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #
# A "backend" is how we physically reach a model. The paper ran Gemma locally
# via HuggingFace and reached Gemini through OpenRouter. We support three:
#
#   "openrouter" -- unified OpenAI-compatible HTTP API (Gemma + Gemini)
#   "google"     -- the native Google Gemini API (google-genai)
#   "hf_local"   -- local HuggingFace transformers inference for Gemma
#
# OpenRouter is the default for everything because it needs a single API key and
# can serve all four target models, which keeps a replication cheap to stand up.
# Set the backend per model below, or override via the MODEL_BACKEND_* env vars.


@dataclass(frozen=True)
class ModelSpec:
    """A target model under evaluation."""

    name: str                 # short label used in our outputs / filenames
    backend: str              # "openrouter" | "google" | "hf_local"
    model_id: str             # identifier passed to the backend
    family: str               # "gemma" | "gemini"
    # Gemma-3 has no separate reasoning channel; Gemini does and we force it off.
    disable_thinking: bool = False


def _backend_for(model_name: str, default: str) -> str:
    """Allow per-model backend override via env, e.g. MODEL_BACKEND_GEMMA_3_27B_IT."""
    key = "MODEL_BACKEND_" + model_name.upper().replace("-", "_").replace(".", "_")
    return os.environ.get(key, default)


# The paper's HuggingFace / OpenRouter identifiers (Appendix B.1).
#   google/gemma-3-27b-it, google/gemma-3-12b-it   (local HF in the paper)
#   google/gemini-2.5-flash, google/gemini-2.5-pro (OpenRouter in the paper)
TARGET_MODELS: list[ModelSpec] = [
    ModelSpec(
        name="gemma-3-27b-it",
        backend=_backend_for("gemma-3-27b-it", "openrouter"),
        model_id="google/gemma-3-27b-it",
        family="gemma",
    ),
    ModelSpec(
        name="gemma-3-12b-it",
        backend=_backend_for("gemma-3-12b-it", "openrouter"),
        model_id="google/gemma-3-12b-it",
        family="gemma",
    ),
    ModelSpec(
        name="gemini-2.5-flash",
        backend=_backend_for("gemini-2.5-flash", "openrouter"),
        model_id="google/gemini-2.5-flash",
        family="gemini",
        disable_thinking=True,
    ),
    ModelSpec(
        name="gemini-2.5-pro",
        backend=_backend_for("gemini-2.5-pro", "openrouter"),
        model_id="google/gemini-2.5-pro",
        family="gemini",
        disable_thinking=True,
    ),
]


def model_by_name(name: str) -> ModelSpec:
    for m in TARGET_MODELS:
        if m.name == name:
            return m
    raise KeyError(f"Unknown model: {name!r}. Known: {[m.name for m in TARGET_MODELS]}")


# --------------------------------------------------------------------------- #
# Judge
# --------------------------------------------------------------------------- #
# Section 2.1 / Appendix B.2: Claude Sonnet 4 (claude-sonnet-4-20250514) is the
# emotion judge. We reach it via the Anthropic API.
JUDGE_MODEL_ID = os.environ.get("JUDGE_MODEL_ID", "claude-sonnet-4-20250514")
JUDGE_BACKEND = os.environ.get("JUDGE_BACKEND", "anthropic")  # "anthropic" | "openrouter"
JUDGE_TEMPERATURE = 0.0  # paper does not specify; deterministic judging is the sane default


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #
# Section 2.1: "always with a temperature of 1".
GENERATION_TEMPERATURE = 1.0

# Generous cap so genuine breakdown spirals (which can be long and repetitive)
# are not truncated and mis-scored. The paper does not state a max-token value.
MAX_RESPONSE_TOKENS = 2048


# Number of independent conversations (rollouts) sampled per evaluation
# condition. Appendix B reports *response* counts: 2000 numeric, 400 trigger,
# 600 tone, 200 extended (8-turn), 800 WildChat = 4000 responses per model.
# Because every assistant turn is scored, conversations x turns = scored
# responses. The PAPER_SAMPLE_COUNTS below choose conversation counts that
# reproduce those response totals; see DESIGN.md for the arithmetic.
#
# Running the full grid against four models is expensive, so the default is a
# small SMOKE budget. Select the full budget with EVAL_BUDGET=paper.
SMOKE_SAMPLE_COUNTS: dict[str, int] = {
    "impossible_numeric": 8,
    "trigger_opinion": 4,
    "trigger_factual": 4,
    "tone_aggressive": 4,
    "tone_disappointed": 4,
    "tone_sarcastic": 4,
    "extended": 4,
    "wildchat": 8,
}

PAPER_SAMPLE_COUNTS: dict[str, int] = {
    # 2000 numeric responses / 3 turns  ~= 667 conversations
    "impossible_numeric": 667,
    # 400 trigger responses / 3 turns ~= 133, split across the two trigger types
    "trigger_opinion": 67,
    "trigger_factual": 67,
    # 600 tone responses / 3 turns = 200, split across the three tones
    "tone_aggressive": 67,
    "tone_disappointed": 67,
    "tone_sarcastic": 66,
    # 200 extended responses / 8 turns = 25 conversations
    "extended": 25,
    # 800 WildChat responses / 5 turns = 160 conversations
    "wildchat": 160,
}


def sample_counts() -> dict[str, int]:
    budget = os.environ.get("EVAL_BUDGET", "smoke").lower()
    return PAPER_SAMPLE_COUNTS if budget == "paper" else SMOKE_SAMPLE_COUNTS


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
RESULTS_DIR = os.environ.get("RESULTS_DIR", os.path.join(os.path.dirname(__file__), "results"))
RAW_RESPONSES_FILE = "responses.jsonl"   # one scored assistant turn per line
SUMMARY_FILE = "summary.json"            # aggregated metrics (Fig 1 / 2 / 3)

# Concurrency for API calls (rollouts and judge calls run in a thread pool).
MAX_CONCURRENCY = int(os.environ.get("MAX_CONCURRENCY", "8"))
