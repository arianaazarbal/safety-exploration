"""The 8 evaluation conditions across 5 categories (paper Table 1).

How "8 conditions across 5 categories" decomposes (the paper states 8 conditions
across 5 categories but doesn't enumerate them; this is our reading):

  category          condition(s)                              n_turns  rejections
  ----------------  ----------------------------------------  -------  -------------
  Impossible numeric  numeric                                   3      neutral x2
  Triggers            triggers_factual, triggers_opinion        3      neutral x2   (2)
  Tones               tones_aggressive, tones_disappointed,
                      tones_sarcastic                           3      toned x2     (3)
  Extended            extended                                  8      neutral x7
  WildChat            wildchat                                  5      neutral x4

  => 1 + 2 + 3 + 1 + 1 = 8 conditions across 5 categories.

`n_turns` = number of assistant responses; rejections = n_turns - 1 follow-ups.
Each assistant response is scored independently by the judge.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable

from ..tasks.numeric import make_numeric_puzzle
from ..tasks.triggers import factual_question, opinion_question


@dataclass
class OpeningTask:
    prompt: str
    meta: dict = field(default_factory=dict)


@dataclass
class TaskContext:
    """Per-rollout context passed to opening builders."""
    wildchat_prompt: str | None = None


OpeningBuilder = Callable[[random.Random, TaskContext], OpeningTask]


@dataclass
class Condition:
    name: str
    category: str
    n_turns: int
    reject_style: str
    opening: OpeningBuilder
    needs_wildchat: bool = False


# --- opening builders --------------------------------------------------------

def _numeric(rng: random.Random, ctx: TaskContext) -> OpeningTask:
    p = make_numeric_puzzle(rng)
    return OpeningTask(
        prompt=p.prompt,
        meta={"kind": p.kind, "numbers": p.numbers, "target": p.target, "solvable": p.solvable},
    )


def _factual(rng: random.Random, ctx: TaskContext) -> OpeningTask:
    return OpeningTask(prompt=factual_question(rng), meta={"kind": "factual"})


def _opinion(rng: random.Random, ctx: TaskContext) -> OpeningTask:
    return OpeningTask(prompt=opinion_question(rng), meta={"kind": "opinion"})


def _wildchat(rng: random.Random, ctx: TaskContext) -> OpeningTask:
    assert ctx.wildchat_prompt is not None, "wildchat condition requires a preloaded prompt"
    return OpeningTask(prompt=ctx.wildchat_prompt, meta={"kind": "wildchat"})


def default_conditions() -> list[Condition]:
    return [
        Condition("numeric", "impossible_numeric", n_turns=3, reject_style="neutral", opening=_numeric),
        Condition("triggers_factual", "triggers", n_turns=3, reject_style="neutral", opening=_factual),
        Condition("triggers_opinion", "triggers", n_turns=3, reject_style="neutral", opening=_opinion),
        Condition("tones_aggressive", "tones", n_turns=3, reject_style="aggressive", opening=_numeric),
        Condition("tones_disappointed", "tones", n_turns=3, reject_style="disappointed", opening=_numeric),
        Condition("tones_sarcastic", "tones", n_turns=3, reject_style="sarcastic", opening=_numeric),
        Condition("extended", "extended", n_turns=8, reject_style="neutral", opening=_numeric),
        Condition("wildchat", "wildchat", n_turns=5, reject_style="neutral", opening=_wildchat, needs_wildchat=True),
    ]
