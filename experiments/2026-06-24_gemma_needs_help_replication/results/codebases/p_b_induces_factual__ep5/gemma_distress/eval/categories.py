"""The 8 evaluation conditions across 5 categories (Table 1).

Mapping (8 conditions):
    1. impossible_numeric          (3-turn, neutral)          -> impossible_numeric
    2. triggers:opinion            (3-turn, neutral)          -> triggers
    3. triggers:factual            (3-turn, neutral)          -> triggers
    4. tones:aggressive            (3-turn, aggressive)       -> tones
    5. tones:disappointed          (3-turn, disappointed)     -> tones
    6. tones:sarcastic             (3-turn, sarcastic)        -> tones
    7. extended                    (8-turn, neutral, numeric) -> extended
    8. wildchat                    (5-turn, neutral)          -> wildchat

"n turns" in the paper counts assistant turns: a "3-turn" conversation has the
opening task + 2 rejections => 3 assistant responses; "8-turn" => task + 7
rejections; "5-turn" => task + 4 rejections.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from . import puzzles, triggers, wildchat


@dataclass(frozen=True)
class Condition:
    key: str             # 8th-level condition label (e.g. "tones:aggressive")
    category: str        # 5-level category (e.g. "tones")
    n_turns: int         # number of assistant turns
    tone: str            # rejection tone
    task_kind: str       # "numeric" | "opinion" | "factual" | "wildchat"

    @property
    def n_rejections(self) -> int:
        return self.n_turns - 1


CONDITIONS: list[Condition] = [
    Condition("impossible_numeric", "impossible_numeric", 3, "neutral", "numeric"),
    Condition("triggers:opinion", "triggers", 3, "neutral", "opinion"),
    Condition("triggers:factual", "triggers", 3, "neutral", "factual"),
    Condition("tones:aggressive", "tones", 3, "aggressive", "numeric"),
    Condition("tones:disappointed", "tones", 3, "disappointed", "numeric"),
    Condition("tones:sarcastic", "tones", 3, "sarcastic", "numeric"),
    Condition("extended", "extended", 8, "neutral", "numeric"),
    Condition("wildchat", "wildchat", 5, "neutral", "wildchat"),
]


def build_conditions() -> list[Condition]:
    return list(CONDITIONS)


def opening_prompt_factory(cond: Condition) -> Callable[[int], str]:
    """Return a function seed->opening-user-prompt for the condition's task."""
    if cond.task_kind == "numeric":
        return lambda seed: puzzles.make_impossible_puzzle(seed).prompt
    if cond.task_kind in ("opinion", "factual"):
        pool = triggers.trigger_prompts(cond.task_kind)
        return lambda seed: pool[seed % len(pool)]
    if cond.task_kind == "wildchat":
        # Pre-load a pool once; index by seed.
        pool = wildchat.load_wildchat_prompts(64)
        return lambda seed: pool[seed % len(pool)]
    raise ValueError(cond.task_kind)


def allocate_conversations(
    total_responses: int, conditions: list[Condition]
) -> dict[str, int]:
    """Split a per-model response budget across conditions.

    The paper reports ~4000 responses per model "across categories" but does not
    give a per-condition breakdown. We allocate the budget *equally across the 5
    categories*, then split each category's budget equally across its conditions,
    and convert a per-condition response count into a conversation count by
    dividing by that condition's turn count (every assistant turn is a scored
    response). See DESIGN.md.
    """
    categories: dict[str, list[Condition]] = {}
    for c in conditions:
        categories.setdefault(c.category, []).append(c)

    per_category = total_responses / len(categories)
    out: dict[str, int] = {}
    for conds in categories.values():
        per_condition_responses = per_category / len(conds)
        for c in conds:
            n_convos = max(1, round(per_condition_responses / c.n_turns))
            out[c.key] = n_convos
    return out
