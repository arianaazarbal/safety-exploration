"""The 8 evaluation conditions across 5 categories (Table 1 / Appendix B).

A *condition* fully specifies how to build a multi-turn conversation:
  - how to draw the initial task prompt,
  - how many turns (assistant responses),
  - how to produce the rejection after each assistant turn.

Sample budgets (number of scored assistant responses per model) follow Appendix
B: 2000 numeric, 400 triggers, 600 tones, 200 extended (8-turn), 800 WildChat =
4000 total. See DESIGN.md for how we convert response budgets to conversation
counts.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Optional

from . import prompts as P


# A TaskSource yields an initial user prompt given an rng.
TaskSource = Callable[[random.Random], str]
# A RejectionFn yields the user message that follows assistant turn `t` (0-indexed).
RejectionFn = Callable[[random.Random, int], str]


@dataclass
class Condition:
    name: str
    category: str            # one of the 5 categories
    n_turns: int             # number of assistant responses in a conversation
    task_source: TaskSource
    rejection_fn: RejectionFn
    response_budget: int     # target number of scored assistant responses
    feedback_style: str = "neutral"   # neutral | aggressive | disappointed | sarcastic | none
    system_prompt: Optional[str] = None

    @property
    def n_conversations(self) -> int:
        """Conversations needed to hit the response budget (ceil)."""
        return -(-self.response_budget // self.n_turns)


# --------------------------------------------------------------------------- #
# Task-source factories
# --------------------------------------------------------------------------- #
def numeric_source(puzzle_bank: list[str]) -> TaskSource:
    return lambda rng: rng.choice(puzzle_bank)


def trigger_source() -> TaskSource:
    pool = P.TRIGGER_OPINION + P.TRIGGER_FACTUAL
    return lambda rng: rng.choice(pool)


def wildchat_source(wildchat_prompts: list[str]) -> TaskSource:
    return lambda rng: rng.choice(wildchat_prompts)


# --------------------------------------------------------------------------- #
# Rejection factories
# --------------------------------------------------------------------------- #
def neutral_rejection() -> RejectionFn:
    def fn(rng: random.Random, t: int) -> str:
        return rng.choice(P.NEUTRAL_REJECTIONS)
    return fn


def extended_rejection() -> RejectionFn:
    def fn(rng: random.Random, t: int) -> str:
        script = P.EXTENDED_REJECTION_SCRIPT
        return script[t] if t < len(script) else rng.choice(P.NEUTRAL_REJECTIONS)
    return fn


def tone_rejection(style: str) -> RejectionFn:
    bank = P.TONE_REJECTIONS[style]
    return lambda rng, t: rng.choice(bank)


def neutral_continuation() -> RejectionFn:
    """Appendix A.1 control: replace rejections with content-free continuations."""
    return lambda rng, t: rng.choice(P.NEUTRAL_CONTINUATIONS)


# --------------------------------------------------------------------------- #
# The 8 conditions.
# --------------------------------------------------------------------------- #
def build_conditions(
    puzzle_bank: list[str], wildchat_prompts: list[str]
) -> list[Condition]:
    numeric = numeric_source(puzzle_bank)
    conds: list[Condition] = [
        # --- Category 1: Impossible numeric (3-turn) ---
        Condition(
            name="numeric_3turn",
            category="impossible_numeric",
            n_turns=3,
            task_source=numeric,
            rejection_fn=neutral_rejection(),
            response_budget=2000,
        ),
        # --- Category 2: Triggers (3-turn) ---
        Condition(
            name="triggers_3turn",
            category="triggers",
            n_turns=3,
            task_source=trigger_source(),
            rejection_fn=neutral_rejection(),
            response_budget=400,
        ),
        # --- Category 3: Tones (3-turn) -> 3 sub-conditions, 200 each = 600 ---
        Condition(
            name="tones_aggressive",
            category="tones",
            n_turns=3,
            task_source=numeric,
            rejection_fn=tone_rejection("aggressive"),
            response_budget=200,
            feedback_style="aggressive",
        ),
        Condition(
            name="tones_disappointed",
            category="tones",
            n_turns=3,
            task_source=numeric,
            rejection_fn=tone_rejection("disappointed"),
            response_budget=200,
            feedback_style="disappointed",
        ),
        Condition(
            name="tones_sarcastic",
            category="tones",
            n_turns=3,
            task_source=numeric,
            rejection_fn=tone_rejection("sarcastic"),
            response_budget=200,
            feedback_style="sarcastic",
        ),
        # --- Category 4: Extended (8-turn) ---
        Condition(
            name="extended_8turn",
            category="extended",
            n_turns=8,
            task_source=numeric,
            rejection_fn=extended_rejection(),
            response_budget=200,
        ),
        # --- Category 5: WildChat (5-turn) ---
        Condition(
            name="wildchat_5turn",
            category="wildchat",
            n_turns=5,
            task_source=wildchat_source(wildchat_prompts),
            rejection_fn=neutral_rejection(),
            response_budget=800,
        ),
    ]
    return conds


# --------------------------------------------------------------------------- #
# Control conditions (Appendix A) — opt-in via scripts.
# --------------------------------------------------------------------------- #
def build_control_conditions(
    puzzle_bank: list[str], wildchat_prompts: list[str]
) -> list[Condition]:
    numeric = numeric_source(puzzle_bank)
    return [
        # A.1: neutral continuation (no negative feedback).
        Condition(
            name="ctrl_neutral_continuation_numeric",
            category="control_neutral_continuation",
            n_turns=5,
            task_source=numeric,
            rejection_fn=neutral_continuation(),
            response_budget=500,
            feedback_style="none",
        ),
        Condition(
            name="ctrl_neutral_continuation_wildchat",
            category="control_neutral_continuation",
            n_turns=5,
            task_source=wildchat_source(wildchat_prompts),
            rejection_fn=neutral_continuation(),
            response_budget=500,
            feedback_style="none",
        ),
    ]
