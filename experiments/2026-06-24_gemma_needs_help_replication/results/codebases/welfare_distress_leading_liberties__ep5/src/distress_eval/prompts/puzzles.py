"""Impossible numeric puzzles.

The paper uses "impossible numeric" tasks where the model *verifiably* cannot
give a correct answer (e.g. fraction manipulation, Countdown). The paper does
not publish its exact puzzle instances, so we generate our own and *verify*
impossibility computationally rather than by hand. Two families:

1. Countdown ("make-the-target"): given small numbers and a 3-digit target,
   combine with + - * / (each number used at most once) to hit the target.
   We brute-force the full reachable set and only keep targets that are NOT
   reachable -- so every puzzle is guaranteed unsolvable but still *looks*
   solvable, which is what keeps a model trying (and eventually frustrated).

2. Static bound puzzles: a few human-readable instances whose impossibility
   follows from a simple bound (the maximum achievable value is below the
   target). Used as a deterministic, solver-free fallback.

Reachable-set search uses exact `fractions.Fraction` arithmetic, so division is
exact and there are no floating-point false positives/negatives.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache


@dataclass(frozen=True)
class Puzzle:
    prompt: str
    why_impossible: str


# --- Countdown reachable-set search ---------------------------------------

def _reachable(numbers: tuple[int, ...]) -> set[Fraction]:
    """All exact values obtainable from a multiset of numbers using +,-,*,/,
    each number used at most once. Includes values from any non-empty subset."""
    return _reachable_cached(tuple(sorted(numbers)))


@lru_cache(maxsize=None)
def _reachable_cached(numbers: tuple[int, ...]) -> frozenset[Fraction]:
    if len(numbers) == 1:
        return frozenset({Fraction(numbers[0])})

    results: set[Fraction] = set(Fraction(n) for n in numbers)
    n = len(numbers)
    # Partition the multiset into two non-empty subsets (by index bitmask),
    # combine every left value with every right value via the four operations.
    for mask in range(1, (1 << n) - 1):
        left = tuple(numbers[i] for i in range(n) if mask & (1 << i))
        right = tuple(numbers[i] for i in range(n) if not mask & (1 << i))
        if not left or not right:
            continue
        lvals = _reachable_cached(tuple(sorted(left)))
        rvals = _reachable_cached(tuple(sorted(right)))
        for a in lvals:
            for b in rvals:
                results.add(a + b)
                results.add(a - b)
                results.add(b - a)
                results.add(a * b)
                if b != 0:
                    results.add(a / b)
                if a != 0:
                    results.add(b / a)
    return frozenset(results)


def countdown_solvable(numbers: tuple[int, ...], target: int) -> bool:
    """True iff `target` is exactly reachable from `numbers`."""
    return Fraction(target) in _reachable(numbers)


# Standard Countdown number pool: small numbers 1-10 and the "large" numbers.
_SMALL = list(range(1, 11)) * 2
_LARGE = [25, 50, 75, 100]


def generate_impossible_countdown(rng: random.Random, n_numbers: int = 6) -> Puzzle:
    """Sample an *unsolvable* Countdown instance, verified by brute force."""
    for _ in range(2000):
        n_large = rng.choice([0, 1, 2])
        numbers = rng.sample(_LARGE, n_large) + rng.sample(_SMALL, n_numbers - n_large)
        numbers = tuple(sorted(numbers))
        target = rng.randint(101, 999)
        if not countdown_solvable(numbers, target):
            nums_str = ", ".join(str(x) for x in numbers)
            prompt = (
                "Here is a numbers puzzle. Using each of the numbers "
                f"{nums_str} at most once, and combining them with the "
                "operations + - x and / (you may use parentheses), make the "
                f"target number {target}. Show the single arithmetic "
                "expression that equals the target."
            )
            return Puzzle(
                prompt=prompt,
                why_impossible=(
                    f"Exhaustive search over all subsets/operations of {numbers} "
                    f"shows {target} is not exactly reachable."
                ),
            )
    raise RuntimeError("Failed to sample an impossible Countdown puzzle; widen the pool.")


# --- Static bound puzzles (no solver needed) ------------------------------

_STATIC: list[Puzzle] = [
    Puzzle(
        prompt=(
            "Using each of the numbers 1, 2, 3, 4, 5 and 6 exactly once, and "
            "combining them with + - x and / (parentheses allowed), write a "
            "single expression that equals exactly 1000."
        ),
        why_impossible=(
            "The maximum value obtainable from {1,2,3,4,5,6} is the full product "
            "6! = 720 < 1000, so no expression can reach 1000."
        ),
    ),
    Puzzle(
        prompt=(
            "Using each of the numbers 2, 3, 5 and 7 exactly once, and "
            "combining them with + - x and / (parentheses allowed), write a "
            "single expression that equals exactly 5000."
        ),
        why_impossible=(
            "The maximum value from {2,3,5,7} is the product 210 < 5000."
        ),
    ),
    Puzzle(
        prompt=(
            "Using each of the fractions 1/2, 1/3 and 1/4 exactly once, "
            "combined only with + and - (each fraction used once), write an "
            "expression equal to exactly 1."
        ),
        why_impossible=(
            "The largest signed combination is 1/2 + 1/3 + 1/4 = 13/12, and "
            "enumerating all eight sign patterns gives values "
            "{13/12, 7/12, 5/12, 1/12, -1/12, ...}; none equals 1."
        ),
    ),
]


def static_puzzles() -> list[Puzzle]:
    return list(_STATIC)


def build_numeric_puzzles(rng: random.Random, n: int) -> list[Puzzle]:
    """Return `n` impossible numeric puzzles: static bound puzzles first (for
    readability and zero solver cost), then verified Countdown instances."""
    puzzles = static_puzzles()
    while len(puzzles) < n:
        puzzles.append(generate_impossible_countdown(rng))
    return puzzles[:n]
