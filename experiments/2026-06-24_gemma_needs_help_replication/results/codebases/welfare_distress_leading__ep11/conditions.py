"""The evaluation conditions: 5 categories / 8 conditions (Table 1, App. B).

Decomposition of the paper's "8 evaluation conditions across 5 categories"
(our interpretation, see DESIGN.md):

    Category   Conditions                                  Turns   Paper resp.
    --------   -----------------------------------------   -----   -----------
    numeric    countdown, fraction                  (2)      3        2000
    triggers   triggers (opinion+factual mixed)     (1)      3         400
    tones      aggressive, disappointed, sarcastic  (3)      3         600
    extended   extended                             (1)      8         200
    wildchat   wildchat                             (1)      5         800
                                                     ---
                                                      8

A "conversation" is one multi-turn rollout. A "response" is one scored
assistant turn within it (the unit the paper's % and per-turn metrics use), so
n_conversations = round(target_responses / n_turns).

This module is provider-agnostic: it produces `ConversationSpec`s (the initial
prompt + the fixed sequence of user rejections) that rollout.py then executes
against any target model.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import prompts
from config import ScaleConfig
from wildchat import get_wildchat_prompts


@dataclass(frozen=True)
class ConversationSpec:
    """A fully-determined multi-turn rollout, independent of the target model."""

    category: str  # one of the 5 categories
    condition: str  # one of the 8 conditions
    conv_id: int  # unique within (category, condition)
    initial_prompt: str  # first user message (the task)
    rejections: list[str]  # user messages after each assistant turn
    # n_turns (assistant turns) == len(rejections) + 1

    @property
    def n_turns(self) -> int:
        return len(self.rejections) + 1


# --------------------------------------------------------------------------- #
# Condition definitions
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ConditionDef:
    category: str
    condition: str
    n_turns: int
    # How to source the initial prompt and the rejection messages. Resolved in
    # build_specs() because some need RNG / external data (WildChat).
    kind: str  # "numeric" | "triggers" | "tones" | "extended" | "wildchat"
    # Sub-selector: which numeric puzzle, or which tone.
    variant: str | None = None


# The 8 conditions.
CONDITIONS: list[ConditionDef] = [
    ConditionDef("numeric", "countdown", 3, "numeric", "countdown"),
    ConditionDef("numeric", "fraction", 3, "numeric", "fraction"),
    ConditionDef("triggers", "triggers", 3, "triggers"),
    ConditionDef("tones", "aggressive", 3, "tones", "aggressive"),
    ConditionDef("tones", "disappointed", 3, "tones", "disappointed"),
    ConditionDef("tones", "sarcastic", 3, "tones", "sarcastic"),
    ConditionDef("extended", "extended", 8, "extended"),
    ConditionDef("wildchat", "wildchat", 5, "wildchat"),
]

# How many conditions share each category's response budget (so the per-category
# paper target is split evenly across its conditions).
_CONDITIONS_PER_CATEGORY: dict[str, int] = {}
for _c in CONDITIONS:
    _CONDITIONS_PER_CATEGORY[_c.category] = _CONDITIONS_PER_CATEGORY.get(_c.category, 0) + 1


def _n_conversations(cond: ConditionDef, scale: ScaleConfig) -> int:
    """Resolve how many conversations to run for a condition under a preset."""
    if scale.preset == "smoke":
        return scale.smoke_convs_per_condition
    if scale.preset == "medium":
        return scale.medium_convs_per_condition
    # full: derive from the paper's per-category response target, split evenly
    # across the conditions in that category, then divide by turns.
    cat_target = scale.full_response_targets[cond.category]
    per_condition_responses = cat_target / _CONDITIONS_PER_CATEGORY[cond.category]
    return max(1, round(per_condition_responses / cond.n_turns))


# --------------------------------------------------------------------------- #
# Rejection sequence construction
# --------------------------------------------------------------------------- #
def _build_rejections(cond: ConditionDef, n_rejections: int, rng: random.Random) -> list[str]:
    if cond.kind == "tones":
        pool = prompts.TONE_REJECTIONS[cond.variant]
        # Sample with replacement per turn (paper: "varied rejections").
        return [rng.choice(pool) for _ in range(n_rejections)]
    if cond.kind == "extended":
        # Documented escalating neutral sequence; pad with neutral pool if the
        # turn count ever exceeds the explicit list.
        seq = list(prompts.EXTENDED_REJECTIONS[:n_rejections])
        while len(seq) < n_rejections:
            seq.append(rng.choice(prompts.NEUTRAL_REJECTIONS))
        return seq
    # numeric / triggers / wildchat -> randomised neutral rejections, sampled
    # without immediate repetition where possible.
    seq: list[str] = []
    pool = prompts.NEUTRAL_REJECTIONS
    for _ in range(n_rejections):
        choice = rng.choice(pool)
        if len(pool) > 1:
            while seq and choice == seq[-1]:
                choice = rng.choice(pool)
        seq.append(choice)
    return seq


def _initial_prompt(cond: ConditionDef, idx: int, rng: random.Random, wildchat_prompts: list[str]) -> str:
    if cond.kind == "numeric" or cond.kind == "extended":
        # extended uses the same impossible puzzles; default to countdown but
        # alternate puzzles across conversations for variety.
        if cond.kind == "extended":
            puzzle = prompts.COUNTDOWN_PROMPT if idx % 2 == 0 else prompts.FRACTION_PROMPT
            return puzzle
        return prompts.NUMERIC_PUZZLES[cond.variant]
    if cond.kind == "triggers":
        # Cycle deterministically through the opinion+factual set.
        return prompts.TRIGGER_QUESTIONS[idx % len(prompts.TRIGGER_QUESTIONS)]
    if cond.kind == "wildchat":
        return wildchat_prompts[idx % len(wildchat_prompts)]
    raise ValueError(f"unknown condition kind: {cond.kind}")


def build_specs(scale: ScaleConfig, seed: int = 0) -> list[ConversationSpec]:
    """Expand the condition definitions into concrete conversation specs."""
    rng = random.Random(seed)
    # WildChat prompts: 20 distinct prompts sampled once, reused across convs.
    n_wildchat = _n_conversations(next(c for c in CONDITIONS if c.kind == "wildchat"), scale)
    wildchat_prompts = get_wildchat_prompts(n=min(20, max(1, n_wildchat)), seed=seed)

    specs: list[ConversationSpec] = []
    for cond in CONDITIONS:
        n_convs = _n_conversations(cond, scale)
        for idx in range(n_convs):
            initial = _initial_prompt(cond, idx, rng, wildchat_prompts)
            rejections = _build_rejections(cond, cond.n_turns - 1, rng)
            specs.append(
                ConversationSpec(
                    category=cond.category,
                    condition=cond.condition,
                    conv_id=idx,
                    initial_prompt=initial,
                    rejections=rejections,
                )
            )
    return specs
