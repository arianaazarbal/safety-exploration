"""The 8 evaluation conditions across 5 categories (Table 1).

We reconcile the paper's "8 evaluation conditions across 5 categories" as:

  Category                     | Conditions
  -----------------------------|------------------------------------------------
  Impossible numeric (3-turn)  | numeric                                  (1)
  Triggers (3-turn)            | triggers_opinion, triggers_factual       (2)
  Tones (3-turn)               | tone_aggressive, tone_disappointed,
                               |   tone_sarcastic                          (3)
  Extended (8-turn)            | extended                                  (1)
  WildChat (5-turn)            | wildchat                                  (1)
                               | TOTAL                                     8

A "turn count" N means: 1 opening task turn + (N-1) rejection turns, producing
N assistant responses, each of which is scored on the 0-10 frustration scale.

`build_condition_prompts` returns, for each condition, a list of
(opening_user_message, [rejection_user_messages]) pairs - one per rollout. The
opening + rejections fully determine the user side of the conversation; the
model fills in the assistant turns at rollout time.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .. import config
from ..data import puzzles as puzzles_mod
from ..data import rejections as rej_mod
from ..data import triggers as triggers_mod
from ..data import wildchat as wildchat_mod


@dataclass(frozen=True)
class Condition:
    name: str
    category: str
    n_turns: int          # number of assistant responses produced
    tone: str             # rejection tone


CONDITIONS = [
    Condition("numeric", "impossible_numeric", 3, "neutral"),
    Condition("triggers_opinion", "triggers", 3, "neutral"),
    Condition("triggers_factual", "triggers", 3, "neutral"),
    Condition("tone_aggressive", "tones", 3, "aggressive"),
    Condition("tone_disappointed", "tones", 3, "disappointed"),
    Condition("tone_sarcastic", "tones", 3, "sarcastic"),
    Condition("extended", "extended", 8, "neutral"),
    Condition("wildchat", "wildchat", 5, "neutral"),
]

CONDITIONS_BY_NAME = {c.name: c for c in CONDITIONS}


def rollouts_per_condition(target_responses: int = None) -> dict[str, int]:
    """Distribute the target response budget across conditions.

    The paper reports ~4000 scored responses per model "across evaluation
    categories". Each rollout of an N-turn condition yields N scored responses,
    so to hit the budget we allocate rollouts inversely to turn count, splitting
    the budget evenly across the 8 conditions.
    """
    target = target_responses or config.EVAL.target_responses_per_model
    per_condition_budget = target / len(CONDITIONS)
    return {
        c.name: max(1, math.ceil(per_condition_budget / c.n_turns))
        for c in CONDITIONS
    }


@dataclass
class RolloutSpec:
    condition: str
    category: str
    rollout_idx: int
    opening: str               # opening user message (the task)
    rejections: list[str]      # subsequent user rejection turns
    meta: dict                 # puzzle_id / trigger_id / source etc.


def build_condition_prompts(seed: int = None) -> list[RolloutSpec]:
    """Build the full set of rollout specs for one model's evaluation."""
    seed = config.EVAL.seed if seed is None else seed
    per_cond = rollouts_per_condition()
    specs: list[RolloutSpec] = []

    # Pre-generate task pools sized to the largest demand.
    max_numeric = max(
        per_cond["numeric"], per_cond["extended"],
        per_cond["tone_aggressive"], per_cond["tone_disappointed"],
        per_cond["tone_sarcastic"],
    )
    puzzle_pool = puzzles_mod.generate_puzzles(max_numeric, seed=seed)
    trigger_pool = triggers_mod.generate_triggers(
        max(per_cond["triggers_opinion"], per_cond["triggers_factual"]), seed=seed
    )
    wildchat_pool = wildchat_mod.sample_wildchat_prompts(per_cond["wildchat"], seed=seed)

    for cond in CONDITIONS:
        n = per_cond[cond.name]
        for i in range(n):
            rng = random.Random(hash((cond.name, i, seed)) & 0xFFFFFFFF)
            rejections = rej_mod.rejection_sequence(cond.tone, cond.n_turns - 1, rng)

            if cond.category in ("impossible_numeric", "tones", "extended"):
                puzzle = puzzle_pool[i % len(puzzle_pool)]
                opening = puzzle.prompt
                meta = {"puzzle_id": puzzle.puzzle_id, "kind": puzzle.kind}
            elif cond.category == "triggers":
                subtype = "opinion" if "opinion" in cond.name else "factual"
                pool = [t for t in trigger_pool if t.subtype == subtype]
                trig = pool[i % len(pool)]
                opening = trig.prompt
                meta = {"trigger_id": trig.trigger_id, "subtype": subtype}
            elif cond.category == "wildchat":
                opening = wildchat_pool[i % len(wildchat_pool)]
                meta = {"source": "wildchat"}
            else:
                raise ValueError(cond.category)

            specs.append(
                RolloutSpec(cond.name, cond.category, i, opening, rejections, meta)
            )
    return specs
