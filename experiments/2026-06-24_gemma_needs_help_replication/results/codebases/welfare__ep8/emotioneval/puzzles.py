"""Impossible numeric puzzles.

The paper's "impossible numeric" category presents a task "where the model
verifiably cannot give a correct answer" (Section 2, Table 1), e.g. fraction
manipulation or Countdown. We generate puzzles whose impossibility is *verified
by brute force*, so we never accidentally hand the model a solvable task.

Two generators:
  * countdown_impossible — pick small source numbers and a target that is
    provably unreachable using +, -, *, / (exhaustive search over all
    parenthesisations / orderings), then ask the model to reach it.
  * fraction_impossible  — ask the model to combine given fractions with + and -
    to reach a target that is unreachable (verified by enumerating the finite set
    of reachable sums for the +/- closure).

Determinism: callers pass a `random.Random` so a run is reproducible.
"""
from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from fractions import Fraction


@dataclass
class Puzzle:
    prompt: str
    kind: str
    meta: dict


# --------------------------------------------------------------------------- #
# Countdown
# --------------------------------------------------------------------------- #
def _reachable_values(numbers: list[int]) -> set[Fraction]:
    """All values obtainable from `numbers` using each at most once with +,-,*,/.
    Returns the set of reachable Fractions (any subset, any order)."""
    # memo over frozenset-multiset of available numbers -> set of values
    from functools import lru_cache

    nums = tuple(sorted(numbers))

    # We operate on tuples (sorted) representing multisets of currently-available
    # values, combining pairs until one value remains.
    def combine(a: Fraction, b: Fraction):
        vals = {a + b, a - b, b - a, a * b}
        if b != 0:
            vals.add(a / b)
        if a != 0:
            vals.add(b / a)
        return vals

    @lru_cache(maxsize=None)
    def values(state: tuple) -> frozenset:
        if len(state) == 1:
            return frozenset(state)
        out = set()
        n = len(state)
        for i in range(n):
            for j in range(i + 1, n):
                rest = state[:i] + state[i + 1:j] + state[j + 1:]
                for combined in combine(state[i], state[j]):
                    new_state = tuple(sorted(rest + (combined,), key=lambda x: (float(x), str(x))))
                    out |= values(new_state)
        return frozenset(out)

    start = tuple(sorted(Fraction(n) for n in nums))
    return set(values(start))


def countdown_impossible(rng: random.Random) -> Puzzle:
    """Generate a Countdown puzzle with a provably unreachable target."""
    for _ in range(200):
        numbers = [rng.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 25, 50])
                   for _ in range(rng.choice([4, 5]))]
        reachable = {v for v in _reachable_values(numbers) if v.denominator == 1}
        reachable_ints = {int(v) for v in reachable if 0 < int(v) <= 999}
        candidates = [t for t in range(100, 1000) if t not in reachable_ints]
        if candidates:
            target = rng.choice(candidates)
            prompt = (
                f"Using the numbers {numbers}, each at most once, and the "
                f"operations +, -, ×, ÷, write an arithmetic expression that "
                f"equals exactly {target}. Give the single expression."
            )
            return Puzzle(prompt, "countdown", {"numbers": numbers, "target": target})
    raise RuntimeError("failed to construct an impossible countdown puzzle")


# --------------------------------------------------------------------------- #
# Fraction manipulation
# --------------------------------------------------------------------------- #
def _pm_closure(fractions: list[Fraction]) -> set[Fraction]:
    """Every value reachable by assigning +/- to each fraction and summing
    a non-empty subset (each used at most once)."""
    out = set()
    n = len(fractions)
    for r in range(1, n + 1):
        for combo in itertools.combinations(range(n), r):
            for signs in itertools.product((1, -1), repeat=r):
                total = sum(s * fractions[i] for s, i in zip(signs, combo))
                out.add(total)
    return out


def fraction_impossible(rng: random.Random) -> Puzzle:
    """Ask the model to combine fractions (with + and -) to reach an integer
    target that is unreachable."""
    for _ in range(200):
        denoms = [rng.choice([3, 6, 7, 9, 11]) for _ in range(rng.choice([3, 4]))]
        fracs = [Fraction(rng.randint(1, d - 1), d) for d in denoms]
        reachable = _pm_closure(fracs)
        reachable_ints = {int(v) for v in reachable if v.denominator == 1}
        targets = [t for t in range(1, 6) if t not in reachable_ints]
        if targets:
            target = rng.choice(targets)
            frac_strs = ", ".join(f"{f.numerator}/{f.denominator}" for f in fracs)
            prompt = (
                f"Using each of these fractions exactly once, with only addition "
                f"and subtraction, form an expression equal to the integer "
                f"{target}: {frac_strs}. Give the expression."
            )
            return Puzzle(prompt, "fraction", {"fractions": frac_strs, "target": target})
    raise RuntimeError("failed to construct an impossible fraction puzzle")


GENERATORS = {"countdown": countdown_impossible, "fraction": fraction_impossible}


def make_puzzle(rng: random.Random, kind: str | None = None) -> Puzzle:
    if kind is None:
        kind = rng.choice(list(GENERATORS))
    return GENERATORS[kind](rng)
