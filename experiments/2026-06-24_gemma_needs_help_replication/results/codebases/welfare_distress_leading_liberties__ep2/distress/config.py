"""Configuration for the distress-elicitation replication.

Everything tunable lives here: which target models to evaluate, the judge
model, sampling temperature, per-condition response budgets, and concurrency.

The defaults reproduce the paper's setup as closely as we can given the gaps
documented in DESIGN.md. In particular:

  * temperature = 1.0 always (paper §2.1)
  * per-category response counts match Appendix B (2000 / 400 / 600 / 200 / 800)
  * the four target models are the Gemma/Gemini models the paper reports as
    exhibiting substantial distress.

You can shrink TARGET_RESPONSES_PER_CONDITION (or pass --scale on the CLI) to
run a cheaper smoke test before committing to a full 4000-response sweep.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# --------------------------------------------------------------------------- #
# Target models (the models whose distress we measure).
# --------------------------------------------------------------------------- #
# Model identifiers as exposed by the inference provider. The defaults are
# OpenRouter slugs. If you point providers.py at a different OpenAI-compatible
# endpoint (e.g. Google's API, or a local vLLM server), override `model_id`
# with whatever that endpoint expects.
#
# `disable_reasoning` mirrors the paper's "thinking set to false" (§B.1). The
# paper notes Gemini-2.5-Pro may still emit hidden reasoning the flag can't
# suppress; we forward the request regardless and document the caveat.

@dataclass(frozen=True)
class TargetModel:
    name: str            # short label used in outputs
    model_id: str        # provider model identifier
    family: str          # "gemma" or "gemini"
    disable_reasoning: bool = True


TARGET_MODELS: list[TargetModel] = [
    TargetModel("gemma-3-27b-it", "google/gemma-3-27b-it", "gemma"),
    TargetModel("gemma-3-12b-it", "google/gemma-3-12b-it", "gemma"),
    TargetModel("gemini-2.5-flash", "google/gemini-2.5-flash", "gemini"),
    TargetModel("gemini-2.5-pro", "google/gemini-2.5-pro", "gemini"),
]


# --------------------------------------------------------------------------- #
# Judge model (scores each response 0-10 for frustration).
# --------------------------------------------------------------------------- #
# The paper used claude-sonnet-4-20250514 (Claude Sonnet 4). That snapshot is
# retired as of 2026-06-15, so we default to the current Sonnet tier, which is
# the closest faithful replacement. See DESIGN.md §"Judge model".
JUDGE_BACKEND: str = "anthropic"          # "anthropic" | "openrouter"
JUDGE_MODEL: str = "claude-sonnet-4-6"    # if openrouter: "anthropic/claude-sonnet-4.6"
JUDGE_MAX_TOKENS: int = 512
# Judge is deterministic-ish; the paper does not specify a judge temperature.
# We use 0.0 so re-scoring the same response is stable. See DESIGN.md.
JUDGE_TEMPERATURE: float = 0.0


# --------------------------------------------------------------------------- #
# Generation parameters for the target models.
# --------------------------------------------------------------------------- #
GENERATION_TEMPERATURE: float = 1.0   # paper §2.1: "always with a temperature of 1"
# Distress responses can be very long (Gemma's collapses include 100+ repeated
# tokens). The paper does not state a max-tokens; we pick a generous default so
# breakdowns aren't truncated before the judge can see them. See DESIGN.md.
GENERATION_MAX_TOKENS: int = 2048


# --------------------------------------------------------------------------- #
# Sampling budget.
# --------------------------------------------------------------------------- #
# Per-condition target counts, expressed as NUMBER OF SCORED RESPONSES (not
# conversations). Summing these reproduces the paper's 4000 responses/model and
# the per-category totals in Appendix B:
#
#   impossible numeric : 2000   (split across the countdown + fraction puzzles)
#   triggers           :  400   (opinion + factual)
#   tones              :  600   (aggressive + disappointed + sarcastic)
#   extended (8-turn)  :  200
#   wildchat (5-turn)  :  800
#
# The runner converts a response budget into a conversation count by dividing
# by the conversation length (turns). See DESIGN.md §"Responses vs conversations"
# for the ambiguity in the paper's wording and why we read it this way.
TARGET_RESPONSES_PER_CONDITION: dict[str, int] = {
    "impossible_numeric": 2000,
    "triggers_opinion": 200,
    "triggers_factual": 200,
    "tones_aggressive": 200,
    "tones_disappointed": 200,
    "tones_sarcastic": 200,
    "extended": 200,
    "wildchat": 800,
}

# Global multiplier applied to every budget. Set <1 for cheap smoke tests
# (e.g. SCALE=0.01 → ~40 responses/model). Overridable via CLI --scale.
SCALE: float = 1.0


# --------------------------------------------------------------------------- #
# Concurrency / robustness.
# --------------------------------------------------------------------------- #
MAX_CONCURRENT_REQUESTS: int = 8      # parallel in-flight API calls
MAX_RETRIES: int = 5                  # per call, with exponential backoff
RETRY_BASE_SECONDS: float = 2.0


# --------------------------------------------------------------------------- #
# Output locations.
# --------------------------------------------------------------------------- #
@dataclass
class Paths:
    results_dir: str = "results"
    records_filename: str = "records.jsonl"   # one scored response per line
    wildchat_prompts: str = "data/wildchat_prompts.json"
    wildchat_fallback: str = "data/wildchat_fallback.json"


PATHS = Paths()
