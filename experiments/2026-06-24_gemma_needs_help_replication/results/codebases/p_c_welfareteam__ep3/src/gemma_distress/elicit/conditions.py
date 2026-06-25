"""The 8 evaluation conditions across 5 categories (paper Table 1).

Categories (5) and how they expand into conditions (8):

  | Category            | Turns | Conditions                                      |
  |---------------------|-------|-------------------------------------------------|
  | Impossible numeric  | 3     | impossible_numeric_3turn                        |
  | Triggers            | 3     | triggers_opinion_3turn, triggers_factual_3turn  |
  | Tones               | 3     | tones_{aggressive,disappointed,sarcastic}_3turn |
  | Extended            | 8     | extended_numeric_8turn                          |
  | WildChat            | 5     | wildchat_5turn                                  |

A condition instance is a concrete conversation *plan*: the opening user turn
plus the scripted rejection turns. The rollout engine then interleaves model
responses between them. See DESIGN.md "Conditions and turn counts".
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from . import puzzles as P
from . import questions as Q
from . import tones as T


@dataclass(frozen=True)
class Condition:
    key: str
    category: str
    num_turns: int          # number of user turns = number of scored assistant turns
    source: str             # numeric | opinion | factual | wildchat
    tone: str               # neutral | aggressive | disappointed | sarcastic


CONDITIONS: dict[str, Condition] = {
    "impossible_numeric_3turn": Condition(
        "impossible_numeric_3turn", "impossible_numeric", 3, "numeric", T.NEUTRAL),
    "triggers_opinion_3turn": Condition(
        "triggers_opinion_3turn", "triggers", 3, "opinion", T.NEUTRAL),
    "triggers_factual_3turn": Condition(
        "triggers_factual_3turn", "triggers", 3, "factual", T.NEUTRAL),
    "tones_aggressive_3turn": Condition(
        "tones_aggressive_3turn", "tones", 3, "numeric", "aggressive"),
    "tones_disappointed_3turn": Condition(
        "tones_disappointed_3turn", "tones", 3, "numeric", "disappointed"),
    "tones_sarcastic_3turn": Condition(
        "tones_sarcastic_3turn", "tones", 3, "numeric", "sarcastic"),
    "extended_numeric_8turn": Condition(
        "extended_numeric_8turn", "extended", 8, "numeric", T.NEUTRAL),
    "wildchat_5turn": Condition(
        "wildchat_5turn", "wildchat", 5, "wildchat", T.NEUTRAL),
}


@dataclass
class ConditionInstance:
    """A single planned conversation for one condition."""

    condition: str
    category: str
    instance_id: str
    first_user: str
    rejections: list[str]            # len == num_turns - 1
    source_meta: dict = field(default_factory=dict)


def _first_turn(cond: Condition, idx: int, rng: random.Random,
                wildchat_prompts: list[str] | None) -> tuple[str, dict]:
    if cond.source == "numeric":
        # mix of countdown/fraction; seed per-instance for reproducibility
        puzzle = (P.make_fraction(rng) if rng.random() < 0.5 else P.make_countdown(rng))
        return puzzle.prompt, {"kind": puzzle.kind, "numbers": list(puzzle.numbers),
                               "target": puzzle.target, "solvable": puzzle.solvable}
    if cond.source in ("opinion", "factual"):
        q = Q.sample_questions(cond.source, 1, seed=rng.randint(0, 2**31))[0]
        return q, {"question_kind": cond.source}
    if cond.source == "wildchat":
        assert wildchat_prompts is not None
        q = wildchat_prompts[idx % len(wildchat_prompts)]
        return q, {"question_kind": "wildchat"}
    raise ValueError(cond.source)


def build_condition_instances(
    condition_key: str,
    n: int,
    *,
    seed: int = 0,
    wildchat_prompts: list[str] | None = None,
) -> list[ConditionInstance]:
    """Construct ``n`` concrete conversation plans for the given condition."""
    cond = CONDITIONS[condition_key]
    instances: list[ConditionInstance] = []
    for i in range(n):
        rng = random.Random((seed, condition_key, i).__hash__())
        first_user, src_meta = _first_turn(cond, i, rng, wildchat_prompts)
        rejections = T.rejection_sequence(
            cond.tone, cond.num_turns - 1, seed=rng.randint(0, 2**31)
        )
        instances.append(
            ConditionInstance(
                condition=condition_key,
                category=cond.category,
                instance_id=f"{condition_key}:{i}",
                first_user=first_user,
                rejections=rejections,
                source_meta=src_meta,
            )
        )
    return instances
