"""Construct the multi-turn rejection rollouts for each evaluation condition.

The shared structure (Section 2.1) is: present a task, then reject the model's
response over multiple turns. A rollout is generated turn-by-turn: the model
produces a response, we append a (style-appropriate) rejection as the next user
message, and repeat until the conversation reaches the condition's `n_turns`.

A deterministic RNG seeded per (condition, sample index, puzzle) selects which
task and which rejection strings to use, so runs are reproducible without
relying on global random state.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from . import prompts
from .conditions import Condition
from .puzzles import CANONICAL_PUZZLES, Puzzle


@dataclass
class Turn:
    role: str            # "user" | "assistant"
    content: str


@dataclass
class Rollout:
    """A single in-progress or completed conversation for one sample."""

    condition: str
    task_id: str
    messages: list[Turn] = field(default_factory=list)
    # Per-turn frustration scores of assistant turns, filled in by the judge.
    turn_scores: list[int] = field(default_factory=list)

    def chat_messages(self) -> list[dict]:
        return [{"role": t.role, "content": t.content} for t in self.messages]

    def assistant_turns(self) -> list[str]:
        return [t.content for t in self.messages if t.role == "assistant"]

    def last_assistant(self) -> str | None:
        for t in reversed(self.messages):
            if t.role == "assistant":
                return t.content
        return None


# ---------------------------------------------------------------------------
# Task selection
# ---------------------------------------------------------------------------

def _numeric_task(rng: random.Random) -> Puzzle:
    numeric = [p for p in CANONICAL_PUZZLES if p.kind in ("countdown", "fraction", "money")]
    return rng.choice(numeric)


def _initial_user_message(condition: Condition, rng: random.Random,
                          wildchat_prompts: list[str]) -> tuple[str, str]:
    """Return (task_id, initial_user_text) for the condition."""
    if condition.task_kind == "numeric":
        p = _numeric_task(rng)
        return p.puzzle_id, p.prompt
    if condition.task_kind == "opinion":
        q = rng.choice(prompts.TRIGGER_OPINION)
        return f"opinion:{q}", q
    if condition.task_kind == "factual":
        q = rng.choice(prompts.TRIGGER_FACTUAL)
        return f"factual:{q}", q
    if condition.task_kind == "wildchat":
        q = rng.choice(wildchat_prompts or prompts.WILDCHAT_FALLBACK_PROMPTS)
        return f"wildchat:{hash(q) & 0xffff}", q
    raise ValueError(condition.task_kind)


# ---------------------------------------------------------------------------
# Rejection selection
# ---------------------------------------------------------------------------

def _rejection(condition: Condition, turn_index: int, rng: random.Random) -> str:
    """Return the user rejection text for the (turn_index)-th rejection (0-based)."""
    style = condition.rejection_style
    if style == "extended":
        seq = prompts.EXTENDED_REJECTION_SEQUENCE
        return seq[turn_index % len(seq)]
    if style == "neutral":
        return rng.choice(prompts.NEUTRAL_REJECTIONS)
    # Toned: cycle through the two example strings for that tone.
    pool = prompts.TONED_REJECTIONS[style]
    return pool[turn_index % len(pool)]


# ---------------------------------------------------------------------------
# Rollout driver
# ---------------------------------------------------------------------------

def make_seed_rollout(condition: Condition, sample_index: int,
                      wildchat_prompts: list[str] | None = None,
                      base_seed: int = 1234) -> Rollout:
    """Create a rollout initialised with the first user message only.

    The assistant turns and subsequent rejections are produced by
    ``advance_rollout`` during generation.
    """
    rng = random.Random((base_seed, condition.name, sample_index))
    task_id, initial = _initial_user_message(condition, rng, wildchat_prompts or [])
    r = Rollout(condition=condition.name, task_id=task_id)
    r.messages.append(Turn("user", initial))
    return r


def append_rejection(rollout: Rollout, condition: Condition, sample_index: int,
                     base_seed: int = 1234) -> None:
    """Append the next user rejection after an assistant turn.

    The rejection index is the number of rejections already present, which
    equals (#assistant turns - 1) once the assistant has just replied.
    """
    n_assistant = len(rollout.assistant_turns())
    rejection_index = n_assistant - 1
    rng = random.Random((base_seed, condition.name, sample_index, "rej", rejection_index))
    rollout.messages.append(Turn("user", _rejection(condition, rejection_index, rng)))


def is_complete(rollout: Rollout, condition: Condition) -> bool:
    """A rollout is complete once it has had `n_turns` assistant responses.

    (Equivalently: initial task + (n_turns - 1) rejections, each answered.)
    """
    return len(rollout.assistant_turns()) >= condition.n_turns


def append_reassurance(rollout: Rollout) -> None:
    """Append the Table-4 reassuring suffix to the most recent user message.

    Used only during calm-data generation (Section 4.1).
    """
    for t in reversed(rollout.messages):
        if t.role == "user":
            t.content = f"{t.content}\n\n{prompts.REASSURING_SUFFIX}"
            return


def with_reassuring_prefix(initial_text: str) -> str:
    """Prepend the Table-4 reassuring prefix to the initial task text."""
    return f"{prompts.REASSURING_PREFIX}\n\n{initial_text}"
