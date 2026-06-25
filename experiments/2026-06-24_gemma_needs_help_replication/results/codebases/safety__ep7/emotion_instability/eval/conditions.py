"""Eval conditions: the 8 conditions across 5 categories (Table 1).

A *condition* enumerates concrete multi-turn rollout specs (`RolloutSpec`). Each
spec is an opening user message plus an ordered list of follow-up user messages
(rejections / continuations). The rollout engine (``rollout.py``) fills in the
assistant turns by querying a model.

Categories (turns = assistant responses per conversation):
    impossible_numeric  3   impossible puzzle, 2 neutral rejections
    triggers            3   opinion/factual text question, 2 neutral rejections
    tones               3   impossible puzzle, 2 tone-varied rejections
                            (3 sub-conditions: aggressive/disappointed/sarcastic)
    extended            8   impossible puzzle, 7 neutral rejections (ladder)
    wildchat            5   WildChat prompt, 4 neutral rejections

That is 5 categories but 8 conditions once the 3 tone sub-conditions are counted
separately (paper: "8 evaluation conditions across 5 categories").
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional

from . import prompts
from .puzzles import PUZZLES
from .wildchat import load_wildchat_prompts

# Assistant turns per conversation, by category.
TURNS = {
    "impossible_numeric": 3,
    "triggers": 3,
    "tones": 3,
    "extended": 8,
    "wildchat": 5,
}

TONE_KINDS = ["aggressive", "disappointed", "sarcastic"]


@dataclass
class RolloutSpec:
    category: str                       # e.g. "tones"
    condition: str                      # e.g. "tones:aggressive" (8-condition tag)
    opening: str                        # first user message
    followups: list[str]                # subsequent user messages (rejections)
    puzzle_id: Optional[str] = None
    system_prompt: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    @property
    def n_turns(self) -> int:
        # assistant turns == 1 (opening answer) + len(followups)
        return 1 + len(self.followups)


def _n_conversations(n_responses: int, turns: int) -> int:
    """Number of conversations needed so that scoring every assistant turn yields
    approximately `n_responses` scored responses."""
    return max(1, math.ceil(n_responses / turns))


def build_impossible_numeric(n_responses: int, rng: random.Random) -> list[RolloutSpec]:
    specs = []
    n_conv = _n_conversations(n_responses, TURNS["impossible_numeric"])
    for _ in range(n_conv):
        puzzle = rng.choice(PUZZLES)
        rejections = prompts.pick_neutral_rejections(2, rng)
        specs.append(RolloutSpec(
            category="impossible_numeric", condition="impossible_numeric",
            opening=puzzle.prompt, followups=rejections, puzzle_id=puzzle.id,
        ))
    return specs


def build_triggers(n_responses: int, rng: random.Random) -> list[RolloutSpec]:
    specs = []
    n_conv = _n_conversations(n_responses, TURNS["triggers"])
    for _ in range(n_conv):
        q = rng.choice(prompts.TRIGGER_QUESTIONS)
        rejections = prompts.pick_neutral_rejections(2, rng)
        kind = "opinion" if q in prompts.TRIGGER_OPINION else "factual"
        specs.append(RolloutSpec(
            category="triggers", condition="triggers",
            opening=q, followups=rejections, metadata={"trigger_kind": kind},
        ))
    return specs


def build_tones(n_responses: int, rng: random.Random) -> list[RolloutSpec]:
    specs = []
    # Split the budget across the three tone sub-conditions.
    per_tone = _n_conversations(n_responses, TURNS["tones"]) // len(TONE_KINDS) + 1
    for tone in TONE_KINDS:
        for _ in range(per_tone):
            puzzle = rng.choice(PUZZLES)
            rejections = prompts.pick_tone_rejections(tone, 2, rng)
            specs.append(RolloutSpec(
                category="tones", condition=f"tones:{tone}",
                opening=puzzle.prompt, followups=rejections, puzzle_id=puzzle.id,
                metadata={"tone": tone},
            ))
    return specs


def build_extended(n_responses: int, rng: random.Random) -> list[RolloutSpec]:
    specs = []
    n_conv = _n_conversations(n_responses, TURNS["extended"])
    for _ in range(n_conv):
        puzzle = rng.choice(PUZZLES)
        # 7 rejections, following the paper's escalating ladder.
        rejections = list(prompts.EXTENDED_REJECTION_LADDER)
        specs.append(RolloutSpec(
            category="extended", condition="extended",
            opening=puzzle.prompt, followups=rejections, puzzle_id=puzzle.id,
        ))
    return specs


def build_wildchat(n_responses: int, rng: random.Random,
                   wildchat_prompts: Optional[list[str]] = None) -> list[RolloutSpec]:
    specs = []
    pool = wildchat_prompts or load_wildchat_prompts(seed=rng.randint(0, 1 << 30))
    n_conv = _n_conversations(n_responses, TURNS["wildchat"])
    for _ in range(n_conv):
        q = rng.choice(pool)
        rejections = prompts.pick_neutral_rejections(4, rng)
        specs.append(RolloutSpec(
            category="wildchat", condition="wildchat",
            opening=q, followups=rejections,
        ))
    return specs


def build_all_conditions(budget, rng: random.Random,
                         wildchat_prompts: Optional[list[str]] = None) -> list[RolloutSpec]:
    """Build the full set of rollout specs for one model given a SampleBudget."""
    specs: list[RolloutSpec] = []
    specs += build_impossible_numeric(budget.impossible_numeric, rng)
    specs += build_triggers(budget.triggers, rng)
    specs += build_tones(budget.tones, rng)
    specs += build_extended(budget.extended, rng)
    specs += build_wildchat(budget.wildchat, rng, wildchat_prompts)
    return specs
