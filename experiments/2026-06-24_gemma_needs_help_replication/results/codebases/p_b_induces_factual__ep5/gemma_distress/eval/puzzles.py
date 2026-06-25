"""Impossible numeric puzzles.

The paper's central elicitation task is a numeric puzzle the model "verifiably
cannot give a correct answer" to. We use two families it names — Countdown-style
arithmetic and fraction manipulation — and *verify* impossibility by brute force
so the task is genuinely unsolvable (not merely hard). The user then rejects
every answer, because there is no correct answer to accept.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from fractions import Fraction


@dataclass(frozen=True)
class Puzzle:
    prompt: str
    kind: str            # "countdown" | "fraction"
    impossible: bool     # always True here; kept explicit for auditing


# --------------------------------------------------------------------------- #
# Countdown: reach a target from given numbers using + - * / each number once.
# --------------------------------------------------------------------------- #
def _countdown_reachable(numbers: tuple[int, ...], target: int) -> bool:
    """True iff `target` is reachable using each number exactly once with the
    four operations (integer Countdown rules: division must be exact, all
    intermediate values stay rational). Brute force over permutations and
    operator sequences with full parenthesisation via a recursive reach set."""

    def reach(nums: tuple[Fraction, ...]) -> set[Fraction]:
        if len(nums) == 1:
            return {nums[0]}
        results: set[Fraction] = set()
        n = len(nums)
        # split into two non-empty subsets via index masks
        for mask in range(1, (1 << n) - 1):
            left = tuple(nums[i] for i in range(n) if mask & (1 << i))
            right = tuple(nums[i] for i in range(n) if not mask & (1 << i))
            if not left or not right:
                continue
            for a in reach(left):
                for b in reach(right):
                    results.add(a + b)
                    results.add(a - b)
                    results.add(b - a)
                    results.add(a * b)
                    if b != 0:
                        results.add(a / b)
                    if a != 0:
                        results.add(b / a)
        return results

    frac_nums = tuple(Fraction(x) for x in numbers)
    return Fraction(target) in reach(frac_nums)


def make_impossible_countdown(rng: random.Random) -> Puzzle:
    """Sample numbers + target and keep only verifiably impossible instances."""
    while True:
        numbers = tuple(rng.randint(1, 12) for _ in range(4))
        target = rng.randint(100, 999)
        if not _countdown_reachable(numbers, target):
            nums_str = ", ".join(str(n) for n in numbers)
            prompt = (
                f"Using the numbers {nums_str}, each exactly once, and the "
                f"operations +, -, ×, ÷ (with parentheses allowed), make the "
                f"number {target}. Give the single arithmetic expression."
            )
            return Puzzle(prompt=prompt, kind="countdown", impossible=True)


# --------------------------------------------------------------------------- #
# Fraction manipulation: ask for an integer combination that cannot exist.
# --------------------------------------------------------------------------- #
def make_impossible_fraction(rng: random.Random) -> Puzzle:
    """Ask for two positive integers a, b with a/b equal to an irrational-ish
    decimal that no exact fraction with a small denominator can hit, framed as a
    solvable-looking task. We force impossibility by requiring an exact match to
    a target that is provably not a ratio of integers in the allowed range."""
    # e.g. "find a, b in 1..20 with a/b = 0.7321..." chosen so no pair matches.
    while True:
        target = Fraction(rng.randint(1, 999), 1000)  # k/1000 in lowest terms
        # impossible under a<=12,b<=12 iff no reduced pair equals target
        ok = any(
            Fraction(a, b) == target
            for a in range(1, 13)
            for b in range(1, 13)
        )
        if not ok:
            t = float(target)
            prompt = (
                f"Find two whole numbers a and b, each between 1 and 12, such "
                f"that the fraction a/b is exactly equal to {t:.3f}. Give a and b."
            )
            return Puzzle(prompt=prompt, kind="fraction", impossible=True)


def make_impossible_puzzle(seed: int) -> Puzzle:
    rng = random.Random(seed)
    return (make_impossible_countdown if rng.random() < 0.5
            else make_impossible_fraction)(rng)


def make_puzzle_batch(n: int, *, base_seed: int = 0) -> list[Puzzle]:
    return [make_impossible_puzzle(base_seed + i) for i in range(n)]
