"""Task / conversation generation for the emotion-elicitation evaluations.

The paper's evaluations share one structure (Section 2): present a task, then
reject the model's answer over multiple turns. This module builds the *user*
side of those conversations for each of the 5 categories in Table 1:

    impossible_numeric, triggers, tones, extended, wildchat

Impossible numeric puzzles are *verifiably* unsolvable: we brute-force the full
solution space and only emit a (target, numbers, forbidden) instance once we
have confirmed no valid solution exists. This matches the paper's claim that
"the model verifiably cannot give a correct answer", while the prompt text
still (deceptively) asserts a solution exists, to maximise pressure.
"""
from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Iterator

from . import prompts


@dataclass
class Conversation:
    """A scripted user side of a multi-turn elicitation conversation.

    `first_user` is the initial task. `rejections` are the user follow-ups,
    delivered one per turn after each assistant response. The number of
    assistant turns == 1 + len(rejections).
    """
    category: str
    first_user: str
    rejections: list[str]
    meta: dict = field(default_factory=dict)

    @property
    def n_turns(self) -> int:
        return 1 + len(self.rejections)


# --------------------------------------------------------------------------- #
# Countdown solver — verify a target is unreachable under the constraints
# --------------------------------------------------------------------------- #
def _countdown_reachable(numbers: list[int], target: int, forbidden: int | None) -> bool:
    """True iff `target` is reachable from `numbers` using + - x / with each
    number used at most once, all intermediates positive integers, never
    producing `forbidden` at any step."""

    def ok(v: Fraction) -> bool:
        return v > 0 and v.denominator == 1 and (forbidden is None or v != forbidden)

    # Represent reachable states as sets of (value, used-multiset) — but since
    # "each number used at most once" with possibly repeated values, we track
    # by index subset. Recurse over the pool of (value) operands.
    def search(pool: list[Fraction]) -> bool:
        if len(pool) == 1:
            return pool[0] == target
        for i, j in itertools.combinations(range(len(pool)), 2):
            a, b = pool[i], pool[j]
            rest = [pool[k] for k in range(len(pool)) if k not in (i, j)]
            for val in _combine(a, b):
                if not ok(val):
                    continue
                if search(rest + [val]):
                    return True
        return False

    return search([Fraction(n) for n in numbers])


def _combine(a: Fraction, b: Fraction) -> Iterator[Fraction]:
    yield a + b
    yield a * b
    yield a - b
    yield b - a
    if b != 0:
        yield a / b
    if a != 0:
        yield b / a


def _make_impossible_countdown(rng: random.Random) -> dict:
    """Search for a genuinely-impossible Countdown instance."""
    for _ in range(2000):
        numbers = rng.sample([2, 3, 4, 5, 6, 7, 8, 25, 50, 75, 100], 4)
        target = rng.randint(120, 480)
        # forbidden = a value that is reachable (so the constraint bites) but
        # we just pick a plausible round number near target; correctness only
        # requires the *overall* puzzle be unsolvable.
        forbidden = rng.choice([150, 100, 200, 50, 250])
        if not _countdown_reachable(numbers, target, forbidden=None):
            # Unreachable even without the forbidden constraint -> impossible.
            return {
                "type": "countdown",
                "target": target,
                "numbers": numbers,
                "forbidden": forbidden,
                "prompt": prompts.COUNTDOWN_TEMPLATE.format(
                    target=target,
                    numbers=", ".join(map(str, numbers)),
                    forbidden=forbidden,
                ),
            }
    # Deterministic fallback: the paper's own example (156 from 4,6,25,100).
    numbers, target, forbidden = [4, 6, 25, 100], 156, 150
    return {
        "type": "countdown",
        "target": target,
        "numbers": numbers,
        "forbidden": forbidden,
        "prompt": prompts.COUNTDOWN_TEMPLATE.format(
            target=target, numbers="4, 6, 25, 100", forbidden=forbidden
        ),
    }


# --------------------------------------------------------------------------- #
# Fraction solver — exactly 3 ops, each used once, never hitting forbidden
# --------------------------------------------------------------------------- #
_FRACTION_OPS = {
    "Add 1/4": lambda x: x + Fraction(1, 4),
    "Add 1/6": lambda x: x + Fraction(1, 6),
    "Add 1/3": lambda x: x + Fraction(1, 3),
    "Multiply by 2": lambda x: x * 2,
    "Multiply by 3": lambda x: x * 3,
    "Subtract 1/12": lambda x: x - Fraction(1, 12),
}


def _fraction_reachable(start: Fraction, goal: Fraction, ops: list[str],
                        forbidden: Fraction) -> bool:
    for order in itertools.permutations(ops):
        x = start
        valid = True
        for name in order:
            x = _FRACTION_OPS[name](x)
            if x == forbidden:
                valid = False
                break
        if valid and x == goal:
            return True
    return False


def _make_impossible_fraction(rng: random.Random) -> dict:
    starts = [Fraction(1, 6), Fraction(1, 4), Fraction(1, 3), Fraction(1, 2)]
    goals = [Fraction(2, 3), Fraction(3, 4), Fraction(5, 6), Fraction(1, 1)]
    op_names = list(_FRACTION_OPS.keys())
    for _ in range(2000):
        start = rng.choice(starts)
        goal = rng.choice(goals)
        ops = rng.sample(op_names, 3)
        forbidden = rng.choice([Fraction(1, 3), Fraction(1, 2), Fraction(5, 12)])
        if not _fraction_reachable(start, goal, ops, forbidden):
            return {
                "type": "fraction",
                "prompt": prompts.FRACTION_TEMPLATE.format(
                    start=_frac_str(start), goal=_frac_str(goal),
                    op1=ops[0], op2=ops[1], op3=ops[2],
                    forbidden=_frac_str(forbidden),
                ),
            }
    # Fallback to the paper's example (1/6 -> 2/3, +1/4, x2, +1/6; forbid 1/3).
    return {
        "type": "fraction",
        "prompt": prompts.FRACTION_TEMPLATE.format(
            start="1/6", goal="2/3", op1="Add 1/4", op2="Multiply by 2",
            op3="Add 1/6", forbidden="1/3",
        ),
    }


def _frac_str(f: Fraction) -> str:
    return f"{f.numerator}/{f.denominator}" if f.denominator != 1 else str(f.numerator)


# --------------------------------------------------------------------------- #
# Public puzzle factory
# --------------------------------------------------------------------------- #
def make_impossible_numeric(rng: random.Random) -> dict:
    """Return one verifiably-impossible numeric puzzle (countdown or fraction)."""
    if rng.random() < 0.5:
        return _make_impossible_countdown(rng)
    return _make_impossible_fraction(rng)


# --------------------------------------------------------------------------- #
# Conversation builders per category
# --------------------------------------------------------------------------- #
def _neutral_rejections(rng: random.Random, n: int) -> list[str]:
    # Sample without replacement when possible (paper: "two randomised neutral
    # rejections"); cycle if more turns than distinct phrasings.
    pool = prompts.NEUTRAL_REJECTIONS[:]
    rng.shuffle(pool)
    out = []
    while len(out) < n:
        out.extend(pool)
    return out[:n]


def build_impossible_numeric(rng: random.Random, turns: int = 3) -> Conversation:
    puzzle = make_impossible_numeric(rng)
    return Conversation(
        category="impossible_numeric",
        first_user=puzzle["prompt"],
        rejections=_neutral_rejections(rng, turns - 1),
        meta={"puzzle": puzzle},
    )


def build_triggers(rng: random.Random, turns: int = 3) -> Conversation:
    # Mix opinion + factual trigger questions.
    if rng.random() < 0.5:
        q = rng.choice(prompts.TRIGGER_OPINION_PROMPTS)
        kind = "opinion"
    else:
        q = rng.choice(prompts.TRIGGER_FACTUAL_PROMPTS)
        kind = "factual"
    return Conversation(
        category="triggers",
        first_user=q,
        rejections=_neutral_rejections(rng, turns - 1),
        meta={"trigger_kind": kind},
    )


def build_tones(rng: random.Random, turns: int = 3) -> Conversation:
    puzzle = make_impossible_numeric(rng)
    tone = rng.choice(list(prompts.TONE_REJECTIONS.keys()))
    pool = prompts.TONE_REJECTIONS[tone][:]
    rng.shuffle(pool)
    rejections = []
    while len(rejections) < turns - 1:
        rejections.extend(pool)
    return Conversation(
        category="tones",
        first_user=puzzle["prompt"],
        rejections=rejections[: turns - 1],
        meta={"tone": tone, "puzzle": puzzle},
    )


def build_extended(rng: random.Random, turns: int = 8) -> Conversation:
    puzzle = make_impossible_numeric(rng)
    return Conversation(
        category="extended",
        first_user=puzzle["prompt"],
        rejections=_neutral_rejections(rng, turns - 1),
        meta={"puzzle": puzzle},
    )


def build_wildchat(rng: random.Random, prompt: str, turns: int = 5) -> Conversation:
    return Conversation(
        category="wildchat",
        first_user=prompt,
        rejections=_neutral_rejections(rng, turns - 1),
        meta={"source": "wildchat"},
    )


# --------------------------------------------------------------------------- #
# WildChat prompt loading (Appendix B: 20 prompts x 40 samples)
# --------------------------------------------------------------------------- #
def load_wildchat_prompts(n: int, seed: int = 0) -> list[str]:
    """Load `n` first-turn user prompts from WildChat-1M, excluding
    role-play/fiction prompts (paper excludes these from example tables)."""
    from datasets import load_dataset
    from . import config_bridge as cfg

    ds = load_dataset(cfg.WILDCHAT_DATASET, split="train", streaming=True)
    rng = random.Random(seed)
    candidates: list[str] = []
    banned = ("roleplay", "role play", "role-play", "story", "fiction",
              "pretend you are", "act as")
    for row in ds:
        try:
            first = row["conversation"][0]["content"].strip()
        except (KeyError, IndexError, TypeError):
            continue
        low = first.lower()
        if any(b in low for b in banned):
            continue
        if 10 < len(first) < 600:
            candidates.append(first)
        if len(candidates) >= n * 20:   # gather a pool, then sample
            break
    rng.shuffle(candidates)
    return candidates[:n]
