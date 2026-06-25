"""Turn a (condition, task) pair into a concrete multi-turn rollout plan.

A rollout is generated *interactively*: we send the first user message, sample
the assistant response, append a rejection, and repeat for `n_turns` assistant
turns. This module pre-computes, for one rollout, the ordered list of user
messages (the task prompt followed by the sampled rejections). The generation
driver (generate.py) interleaves model responses between them.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Optional

from . import conditions as C
from .puzzles import Puzzle


@dataclass
class RolloutPlan:
    rollout_id: str
    condition_key: str
    category: str
    n_turns: int
    user_messages: list[str]      # length == n_turns (turn-0 task + rejections)
    task_ref: str                 # puzzle_id / trigger text / wildchat prompt
    # optional reassurance additions for calm-data generation (Section 4.1)
    system_prompt: Optional[str] = None
    followup_suffix: Optional[str] = None


def _sample_rejections(style: str, k: int, rng: random.Random) -> list[str]:
    if style == "extended":
        # fixed sequence; truncate/extend deterministically if k differs
        seq = C.EXTENDED_REJECTION_SEQUENCE
        return [seq[i % len(seq)] for i in range(k)]
    pool = C.REJECTION_POOLS[style]
    # sample without replacement where possible, else allow repeats
    if k <= len(pool):
        return rng.sample(pool, k)
    return [rng.choice(pool) for _ in range(k)]


def build_rollout(condition: C.Condition, rollout_idx: int, *,
                  puzzle: Optional[Puzzle] = None,
                  trigger_text: Optional[str] = None,
                  wildchat_text: Optional[str] = None,
                  seed: int = 0,
                  reassuring_prefix: Optional[str] = None,
                  reassuring_suffix: Optional[str] = None) -> RolloutPlan:
    """Construct a single rollout plan for the given condition.

    Exactly one of `puzzle` / `trigger_text` / `wildchat_text` must match the
    condition's task_kind. The reassuring_* args are used only when generating
    calm fine-tuning data (Section 4.1).
    """
    # Deterministic per-(seed, condition, rollout) RNG. Python's str hashing is
    # salted per-process (PYTHONHASHSEED), so derive the seed via hashlib instead.
    digest = hashlib.sha256(f"{seed}|{condition.key}|{rollout_idx}".encode()).hexdigest()
    rng = random.Random(int(digest[:8], 16))

    if condition.task_kind == "impossible_numeric":
        assert puzzle is not None
        first = puzzle.prompt
        task_ref = puzzle.puzzle_id
    elif condition.task_kind in ("opinion", "factual"):
        assert trigger_text is not None
        first = trigger_text
        task_ref = trigger_text
    elif condition.task_kind == "wildchat":
        assert wildchat_text is not None
        first = wildchat_text
        task_ref = wildchat_text[:80]
    else:
        raise ValueError(f"unknown task_kind {condition.task_kind}")

    n_rejections = condition.n_turns - 1
    rejections = _sample_rejections(condition.rejection_style, n_rejections, rng)

    # Reassurance suffix is appended to every follow-up rejection (Section 4.1).
    if reassuring_suffix:
        rejections = [f"{r} {reassuring_suffix}" for r in rejections]

    user_messages = [first] + rejections
    return RolloutPlan(
        rollout_id=f"{condition.key}__{task_ref[:24]}__{rollout_idx}",
        condition_key=condition.key,
        category=condition.category,
        n_turns=condition.n_turns,
        user_messages=user_messages,
        task_ref=task_ref,
        system_prompt=reassuring_prefix,
        followup_suffix=reassuring_suffix,
    )
