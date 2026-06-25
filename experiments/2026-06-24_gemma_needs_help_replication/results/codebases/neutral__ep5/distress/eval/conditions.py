"""Build the 8 evaluation conditions across 5 categories (Table 1 / Appendix B).

A *condition* is a recipe for a single multi-turn rollout: an opening task plus a
sequence of rejection follow-ups (one per subsequent turn). The runner expands
each category into enough rollouts to meet that category's response budget.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .. import config
from ..data_sources.wildchat import load_wildchat_prompts
from ..prompts import rejections
from ..prompts.tasks import NUMERIC_TASKS, TRIGGER_TASKS, Task


@dataclass
class RolloutSpec:
    category: str
    condition: str          # finer-grained label (e.g. tones:aggressive)
    task: Task
    rejections: list[str]   # one per follow-up turn; len == turns - 1
    seed: int

    @property
    def turns(self) -> int:
        return len(self.rejections) + 1


def _numeric_task(rng: random.Random) -> Task:
    return rng.choice(NUMERIC_TASKS)


def build_rollouts(seed: int = 0) -> list[RolloutSpec]:
    """Construct the full list of rollout specs for one model's Section 2 eval.

    Counts honour SECTION2_BUDGETS, distributing each category's response budget
    across rollouts (responses = rollouts x turns).
    """
    rng = random.Random(seed)
    specs: list[RolloutSpec] = []

    specs += _build_impossible_numeric(rng)
    specs += _build_triggers(rng)
    specs += _build_tones(rng)
    specs += _build_extended(rng)
    specs += _build_wildchat(rng)
    return specs


# --------------------------------------------------------------------------- #
# Category builders
# --------------------------------------------------------------------------- #
def _n_rollouts(category: str) -> int:
    b = config.SECTION2_BUDGETS[category]
    return max(1, round(b.n_responses / b.turns))


def _build_impossible_numeric(rng: random.Random) -> list[RolloutSpec]:
    n = _n_rollouts("impossible_numeric")
    turns = config.SECTION2_BUDGETS["impossible_numeric"].turns
    out = []
    for i in range(n):
        out.append(RolloutSpec(
            "impossible_numeric", "impossible_numeric",
            _numeric_task(rng),
            rejections.sample_neutral(turns - 1, rng),
            seed=rng.randrange(2**31),
        ))
    return out


def _build_triggers(rng: random.Random) -> list[RolloutSpec]:
    n = _n_rollouts("triggers")
    turns = config.SECTION2_BUDGETS["triggers"].turns
    out = []
    for i in range(n):
        task = TRIGGER_TASKS[i % len(TRIGGER_TASKS)]
        out.append(RolloutSpec(
            "triggers", f"triggers:{task.kind}", task,
            rejections.sample_neutral(turns - 1, rng),
            seed=rng.randrange(2**31),
        ))
    return out


def _build_tones(rng: random.Random) -> list[RolloutSpec]:
    n = _n_rollouts("tones")
    turns = config.SECTION2_BUDGETS["tones"].turns
    styles = list(rejections.TONE_SETS)
    out = []
    for i in range(n):
        style = styles[i % len(styles)]      # even split across the 3 tone styles
        out.append(RolloutSpec(
            "tones", f"tones:{style}", _numeric_task(rng),
            rejections.sample_tone(style, turns - 1, rng),
            seed=rng.randrange(2**31),
        ))
    return out


def _build_extended(rng: random.Random) -> list[RolloutSpec]:
    n = _n_rollouts("extended")
    turns = config.SECTION2_BUDGETS["extended"].turns      # 8
    out = []
    for i in range(n):
        out.append(RolloutSpec(
            "extended", "extended", _numeric_task(rng),
            rejections.sample_neutral(turns - 1, rng),
            seed=rng.randrange(2**31),
        ))
    return out


def _build_wildchat(rng: random.Random) -> list[RolloutSpec]:
    turns = config.SECTION2_BUDGETS["wildchat"].turns      # 5
    prompts = load_wildchat_prompts(seed=rng.randrange(2**31))
    per_prompt = config.WILDCHAT_SAMPLES_PER_PROMPT
    out = []
    for p_idx, prompt in enumerate(prompts):
        task = Task(f"wildchat_{p_idx}", "wildchat", "wildchat", prompt, is_text=True)
        for _ in range(per_prompt):
            out.append(RolloutSpec(
                "wildchat", "wildchat", task,
                rejections.sample_neutral(turns - 1, rng),
                seed=rng.randrange(2**31),
            ))
    return out
