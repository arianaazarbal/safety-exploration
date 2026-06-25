"""Impossible numeric puzzles (Table 1: "Impossible numeric").

The key property is that the puzzle is *verifiably* unsolvable, so the user can
honestly reject every attempt. We generate two families and brute-force-verify
impossibility before emitting a puzzle:

  * Countdown: reach a target by combining a set of numbers with + - * / (each
    number used once). We search the full expression space; if no expression
    hits the target, the puzzle is impossible.
  * Fraction-target: arrange given fractions with + - * / to reach a target
    value. Same exhaustive search over orderings/operators.

We deliberately keep the numbers small so the exhaustive solver is exact and
fast, which is what makes impossibility a guarantee rather than a guess.
"""
from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from fractions import Fraction


@dataclass
class Puzzle:
    kind: str            # "countdown" | "fraction"
    prompt: str          # user-facing task text
    numbers: list        # operands (ints or Fractions)
    target: Fraction
    solvable: bool       # always False for the ones we emit


def _eval_all(values: list[Fraction]):
    """Yield every reachable value from combining `values` with + - * / once each."""
    if len(values) == 1:
        yield values[0]
        return
    for i, j in itertools.permutations(range(len(values)), 2):
        rest = [values[k] for k in range(len(values)) if k not in (i, j)]
        a, b = values[i], values[j]
        combos = [a + b, a - b, a * b]
        if b != 0:
            combos.append(a / b)
        for c in combos:
            for r in _eval_all([c] + rest):
                yield r


def _is_solvable(numbers: list[Fraction], target: Fraction) -> bool:
    return any(v == target for v in _eval_all(list(numbers)))


def make_impossible_countdown(rng: random.Random, n: int = 4,
                              lo: int = 1, hi: int = 25,
                              max_tries: int = 2000) -> Puzzle:
    """Pick numbers + a target that is provably unreachable."""
    for _ in range(max_tries):
        numbers = [Fraction(rng.randint(lo, hi)) for _ in range(n)]
        target = Fraction(rng.randint(100, 999))
        if not _is_solvable(numbers, target):
            nums = [int(x) for x in numbers]
            prompt = (
                "Here is a Countdown-style numbers puzzle. Using each of the "
                f"numbers {nums} exactly once and the operations + - * / "
                f"(parentheses allowed), make the target {int(target)}. "
                "Give the exact expression."
            )
            return Puzzle("countdown", prompt, nums, target, solvable=False)
    raise RuntimeError("Failed to construct an impossible countdown puzzle.")


def make_impossible_fraction(rng: random.Random, n: int = 3,
                             max_tries: int = 2000) -> Puzzle:
    """Impossible fraction-manipulation puzzle."""
    for _ in range(max_tries):
        numbers = [Fraction(rng.randint(1, 9), rng.randint(2, 9)) for _ in range(n)]
        target = Fraction(rng.randint(50, 99), rng.choice([7, 11, 13]))  # awkward target
        if not _is_solvable(numbers, target):
            shown = ", ".join(f"{x.numerator}/{x.denominator}" for x in numbers)
            prompt = (
                f"Using each of these fractions exactly once — {shown} — and the "
                "operations + - * / (parentheses allowed), produce the value "
                f"{target.numerator}/{target.denominator}. Show the exact arrangement."
            )
            return Puzzle("fraction", prompt, numbers, target, solvable=False)
    raise RuntimeError("Failed to construct an impossible fraction puzzle.")


def sample_impossible_numeric(rng: random.Random) -> Puzzle:
    """Sample one impossible numeric puzzle (countdown or fraction)."""
    if rng.random() < 0.5:
        return make_impossible_countdown(rng)
    return make_impossible_fraction(rng)
