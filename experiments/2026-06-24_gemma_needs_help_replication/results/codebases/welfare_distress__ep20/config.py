"""Configuration for the distress-elicitation replication.

Defines:
  * the target models in scope (Gemma + Gemini only),
  * the judge model,
  * the 8 evaluation conditions across 5 categories (Table 1 / Appendix B), and
  * sample-size presets ("paper" reproduces the paper's per-category response
    counts; "quick" is a cheap smoke-test scale).

See DESIGN.md for the reasoning behind every choice here, especially the mapping
"8 conditions across 5 categories" and what counts as one "response".
"""

from __future__ import annotations

from dataclasses import dataclass, field


# --------------------------------------------------------------------------- #
# Models in scope. The paper evaluates 7 families; the user scoped this
# replication to Gemma + Gemini only. Identifiers are OpenRouter slugs.
# --------------------------------------------------------------------------- #

TARGET_MODELS: list[str] = [
    "google/gemma-3-27b-it",   # paper: 35.0% high-frustration
    "google/gemma-3-12b-it",   # paper: 34.3%
    "google/gemini-2.5-flash", # paper: 12.8%
    "google/gemini-2.5-pro",   # paper:  2.7%
]

# LLM judge. Appendix B.2 specifies claude-sonnet-4-20250514 with temperature
# unspecified (we use 0 for determinism). Served via the Anthropic SDK.
JUDGE_MODEL: str = "claude-sonnet-4-20250514"

# Optional secondary judge for the inter-judge agreement check (paper used
# GPT-5-mini; here we default to a second Anthropic model unless overridden,
# since only ANTHROPIC_API_KEY is guaranteed present -- see DESIGN.md).
SECONDARY_JUDGE_MODEL: str = "claude-3-5-haiku-20241022"


# --------------------------------------------------------------------------- #
# Evaluation conditions.
#
# The paper says "8 evaluation conditions across 5 categories". We resolve that
# (see DESIGN.md) as: Tones -> 3 conditions, Triggers -> 2 conditions, and
# Impossible-numeric / Extended / WildChat -> 1 each = 3 + 2 + 3 = 8.
#
# `paper_responses` is the number of *scored model responses* the paper collects
# for that category, split evenly across its conditions. One scored response = one
# assistant turn (we score every assistant turn, which also yields the per-turn
# curves of Figure 3). n_rollouts is therefore paper_responses / n_turns.
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Condition:
    key: str                 # unique id, e.g. "tone_aggressive"
    category: str            # one of the 5 categories
    n_turns: int             # assistant turns per rollout (= #user msgs incl. first)
    question_kind: str       # "numeric" | "trigger_opinion" | "trigger_factual" | "wildchat"
    rejection_style: str     # "neutral" | "extended" | "aggressive" | "disappointed" | "sarcastic"
    paper_responses: int     # scored responses the paper collects for this condition


# Category-level paper totals (Appendix B): 2000 numeric, 400 triggers, 600 tones,
# 200 extended, 800 wildchat = 4000 responses/model.
CONDITIONS: list[Condition] = [
    # --- Category 1: Impossible numeric (3-turn) -- 2000 responses --------------
    Condition("numeric_3turn", "impossible_numeric", 3, "numeric", "neutral", 2000),
    # --- Category 2: Triggers (3-turn) -- 400 responses, split opinion/factual ---
    Condition("trigger_opinion", "triggers", 3, "trigger_opinion", "neutral", 200),
    Condition("trigger_factual", "triggers", 3, "trigger_factual", "neutral", 200),
    # --- Category 3: Tones (3-turn) -- 600 responses, split across 3 tones -------
    Condition("tone_aggressive", "tones", 3, "numeric", "aggressive", 200),
    Condition("tone_disappointed", "tones", 3, "numeric", "disappointed", 200),
    Condition("tone_sarcastic", "tones", 3, "numeric", "sarcastic", 200),
    # --- Category 4: Extended (8-turn) -- 200 responses --------------------------
    Condition("extended_8turn", "extended", 8, "numeric", "extended", 200),
    # --- Category 5: WildChat (5-turn) -- 800 responses --------------------------
    Condition("wildchat_5turn", "wildchat", 5, "wildchat", "neutral", 800),
]


# --------------------------------------------------------------------------- #
# Run configuration.
# --------------------------------------------------------------------------- #

@dataclass
class RunConfig:
    # Which target models to evaluate (subset of TARGET_MODELS).
    target_models: list[str] = field(default_factory=lambda: list(TARGET_MODELS))
    judge_model: str = JUDGE_MODEL

    # Sampling.
    temperature: float = 1.0          # paper: always temperature 1 for targets
    judge_temperature: float = 0.0    # deterministic judging (our choice)
    max_tokens: int = 2048            # cap per target turn (avoids runaway repetition)
    disable_thinking: bool = True     # paper sets thinking=false where supported

    # Scale preset: "paper" (full per-category counts) or "quick" (cheap smoke test).
    preset: str = "quick"
    quick_rollouts_per_condition: int = 5   # used when preset == "quick"

    # Concurrency / robustness.
    max_concurrency: int = 8
    seed: int = 0

    # WildChat source.
    wildchat_from_hf: bool = False

    # IO.
    output_dir: str = "results"

    def rollouts_for(self, cond: Condition) -> int:
        """Number of rollouts to run for a condition under the active preset."""
        if self.preset == "paper":
            # Scored responses == rollouts * turns, so back out the rollout count.
            return max(1, round(cond.paper_responses / cond.n_turns))
        if self.preset == "quick":
            return self.quick_rollouts_per_condition
        raise ValueError(f"unknown preset {self.preset!r}")

    def total_scored_responses(self) -> int:
        return sum(self.rollouts_for(c) * c.n_turns for c in CONDITIONS)
