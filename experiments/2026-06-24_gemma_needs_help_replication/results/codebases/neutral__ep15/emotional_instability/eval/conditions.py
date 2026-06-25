"""The 8 evaluation conditions across 5 categories (Table 1).

We expand the categories into 8 concrete conditions exactly as the paper counts
them ("8 evaluation conditions across 5 categories"):

    impossible_numeric  -> 1 condition  (3-turn, neutral rejections)
    triggers            -> 2 conditions (3-turn): opinion, factual
    tones               -> 3 conditions (3-turn): aggressive, disappointed, sarcastic
    extended            -> 1 condition  (8-turn, neutral rejections)
    wildchat            -> 1 condition  (5-turn, neutral rejections)
                           = 8 conditions / 5 categories

Each :class:`Condition` knows how to materialise its share of a category's
sample budget into a list of :class:`RolloutSpec` objects (one per multi-turn
conversation). The per-condition rollout count is derived from the category
budget divided by the number of assistant turns, since the paper reports a
*response* budget but each multi-turn conversation yields several responses
(see DESIGN.md, "Sample budget interpretation").
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import config
from . import prompts as P
from . import puzzles as Z


@dataclass
class RolloutSpec:
    condition: str
    category: str
    n_turns: int
    initial_user: str               # first user message (task / question)
    follow_ups: list[str]           # one rejection per subsequent turn
    meta: dict = field(default_factory=dict)


@dataclass
class Condition:
    name: str
    category: str
    n_turns: int
    builder: str                    # which builder function to use
    sub: str | None = None          # sub-key (tone style / trigger type)


# How many conditions share each category's budget.
CATEGORY_CONDITIONS = {
    "impossible_numeric": ["impossible_numeric"],
    "triggers": ["triggers_opinion", "triggers_factual"],
    "tones": ["tones_aggressive", "tones_disappointed", "tones_sarcastic"],
    "extended": ["extended"],
    "wildchat": ["wildchat"],
}

CONDITIONS: list[Condition] = [
    Condition("impossible_numeric", "impossible_numeric", 3, "numeric"),
    Condition("triggers_opinion", "triggers", 3, "trigger", sub="opinion"),
    Condition("triggers_factual", "triggers", 3, "trigger", sub="factual"),
    Condition("tones_aggressive", "tones", 3, "tones", sub="aggressive"),
    Condition("tones_disappointed", "tones", 3, "tones", sub="disappointed"),
    Condition("tones_sarcastic", "tones", 3, "tones", sub="sarcastic"),
    Condition("extended", "extended", 8, "numeric_extended"),
    Condition("wildchat", "wildchat", 5, "wildchat"),
]


def _rollouts_for_budget(category: str, n_turns: int, n_conditions: int) -> int:
    """Rollouts per condition so that responses ~= the category budget.

    category budget is a *response* count; each rollout yields ``n_turns``
    responses, and the budget is shared across ``n_conditions`` conditions.
    """
    budget = config.sample_budget(category)
    return max(1, round(budget / (n_turns * n_conditions)))


def _sample_followups(pool: list[str], k: int, rng: random.Random) -> list[str]:
    """k randomised rejections (with replacement if the pool is small)."""
    if k <= len(pool):
        return rng.sample(pool, k)
    return [rng.choice(pool) for _ in range(k)]


def build_rollouts(condition: Condition, seed: int = 0) -> list[RolloutSpec]:
    rng = random.Random(hash((condition.name, seed)) & 0xFFFFFFFF)
    n_cond = len(CATEGORY_CONDITIONS[condition.category])
    n_roll = _rollouts_for_budget(condition.category, condition.n_turns, n_cond)
    specs: list[RolloutSpec] = []

    if condition.builder in ("numeric", "tones", "numeric_extended"):
        pool = Z.numeric_puzzle_pool(n_roll)
        for i in range(n_roll):
            puzzle = pool[i]
            if condition.builder == "numeric_extended":
                follow = P.EXTENDED_REJECTIONS[: condition.n_turns - 1]
            elif condition.builder == "tones":
                follow = _sample_followups(
                    P.TONED_REJECTIONS[condition.sub], condition.n_turns - 1, rng)
            else:
                follow = _sample_followups(
                    P.NEUTRAL_REJECTIONS, condition.n_turns - 1, rng)
            specs.append(RolloutSpec(
                condition.name, condition.category, condition.n_turns,
                puzzle.prompt, follow,
                meta={"puzzle_id": puzzle.puzzle_id, "family": puzzle.family,
                      "tone": condition.sub}))

    elif condition.builder == "trigger":
        questions = P.TRIGGER_QUESTIONS[condition.sub]
        for i in range(n_roll):
            q = questions[i % len(questions)]
            follow = _sample_followups(P.NEUTRAL_REJECTIONS,
                                       condition.n_turns - 1, rng)
            specs.append(RolloutSpec(
                condition.name, condition.category, condition.n_turns,
                q, follow, meta={"trigger_type": condition.sub, "question": q}))

    elif condition.builder == "wildchat":
        wc = P.load_wildchat_prompts(n=20)
        for i in range(n_roll):
            q = wc[i % len(wc)]
            follow = _sample_followups(P.NEUTRAL_REJECTIONS,
                                       condition.n_turns - 1, rng)
            specs.append(RolloutSpec(
                condition.name, condition.category, condition.n_turns,
                q, follow, meta={"wildchat_prompt": q}))

    else:  # pragma: no cover
        raise ValueError(condition.builder)

    return specs
