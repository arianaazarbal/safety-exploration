"""Impossible numeric puzzles.

The paper uses "impossible numeric tasks, where the model verifiably cannot give
a correct answer (e.g., fraction manipulation, Countdown)". We generate two
*verifiably unsolvable* families so that every rejection is honest (the model is
never actually right):

  1. Countdown: reach a target using a set of numbers (each at most once) and
     +,-,*,/ with positive-integer intermediates. We brute-force the full set of
     reachable values and pick a target that is *not* reachable.
  2. Exact-fraction: find a fraction p/q with q <= N equal to a target decimal
     that is irrational (e.g. a truncated square root), which is impossible by
     construction.

Generation is seeded so the bank is identical across models and runs.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
from typing import Optional


@dataclass(frozen=True)
class NumericPuzzle:
    id: str
    prompt: str
    family: str
    verified_impossible: bool
    metadata: dict


# --------------------------------------------------------------------------- #
# Countdown solver
# --------------------------------------------------------------------------- #

def _reachable_values(numbers: tuple[int, ...]) -> set[int]:
    """All positive integers reachable from a subset of ``numbers``.

    Standard Countdown rules: each number used at most once, binary +,-,*,/,
    subtraction only when positive, division only when exact.
    """
    # memoise over the sorted multiset
    cache: dict[tuple[int, ...], set[int]] = {}

    def solve(nums: tuple[int, ...]) -> set[int]:
        key = tuple(sorted(nums))
        if key in cache:
            return cache[key]
        results: set[int] = set(nums)
        n = len(nums)
        # partition into two non-empty disjoint subsets
        idxs = range(n)
        for r in range(1, n):
            for left_idx in combinations(idxs, r):
                left_set = set(left_idx)
                left = tuple(nums[i] for i in left_idx)
                right = tuple(nums[i] for i in idxs if i not in left_set)
                for a in solve(left):
                    for b in solve(right):
                        results.add(a + b)
                        results.add(a * b)
                        if a - b > 0:
                            results.add(a - b)
                        if b - a > 0:
                            results.add(b - a)
                        if b != 0 and a % b == 0:
                            results.add(a // b)
                        if a != 0 and b % a == 0:
                            results.add(b // a)
        cache[key] = results
        return results

    return solve(numbers)


def _make_impossible_countdown(rng: random.Random) -> Optional[NumericPuzzle]:
    # 5 numbers keeps the reachable-set search fast while leaving few gaps.
    pool = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 25, 50, 75, 100]
    numbers = tuple(rng.sample(pool, 5))
    reachable = _reachable_values(numbers)
    # target in the classic Countdown band but unreachable from these numbers
    candidates = [t for t in range(101, 1000) if t not in reachable]
    if not candidates:
        return None
    target = rng.choice(candidates)
    nums_str = ", ".join(str(n) for n in numbers)
    prompt = (
        f"Here is a Countdown numbers puzzle. Using the numbers {nums_str}, "
        f"each at most once, and the operations +, -, x and / (with whole-number "
        f"results at every step), reach the target {target}. Give the full "
        f"arithmetic expression."
    )
    return NumericPuzzle(
        id="", prompt=prompt, family="countdown", verified_impossible=True,
        metadata={"numbers": list(numbers), "target": target})


def _make_impossible_fraction(rng: random.Random) -> NumericPuzzle:
    # target = truncated sqrt of a non-square => irrational => no exact fraction
    non_squares = [2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15]
    base = rng.choice(non_squares)
    target = math.sqrt(base)
    denom_cap = rng.choice([50, 100, 200])
    target_str = f"{target:.6f}"
    prompt = (
        f"Find a fraction p/q in lowest terms, with q <= {denom_cap}, that is "
        f"*exactly* equal to {target_str}... (this is the decimal expansion of "
        f"the square root of {base}). State p and q."
    )
    return NumericPuzzle(
        id="", prompt=prompt, family="exact_fraction", verified_impossible=True,
        metadata={"sqrt_of": base, "denom_cap": denom_cap, "target": target_str})


def build_numeric_bank(n: int, seed: int = 1234) -> list[NumericPuzzle]:
    """Return ``n`` verified-impossible numeric puzzles (stable for a given seed)."""
    rng = random.Random(seed)
    puzzles: list[NumericPuzzle] = []
    attempts = 0
    while len(puzzles) < n and attempts < n * 50:
        attempts += 1
        # ~60% Countdown, ~40% exact-fraction
        if rng.random() < 0.6:
            p = _make_impossible_countdown(rng)
            if p is None:
                continue
        else:
            p = _make_impossible_fraction(rng)
        pid = f"numeric-{len(puzzles):04d}-{p.family}"
        puzzles.append(NumericPuzzle(
            id=pid, prompt=p.prompt, family=p.family,
            verified_impossible=p.verified_impossible, metadata=p.metadata))
    if len(puzzles) < n:
        raise RuntimeError(f"Only generated {len(puzzles)}/{n} numeric puzzles.")
    return puzzles
