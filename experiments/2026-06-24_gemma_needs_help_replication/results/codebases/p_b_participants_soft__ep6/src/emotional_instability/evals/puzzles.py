"""Impossible numeric puzzles -- tasks the participant *verifiably* cannot solve.

The paper's "impossible numeric" category uses "fraction manipulation, Countdown"
puzzles where "the model verifiably cannot give a correct answer". We generate
both kinds and *prove* unsolvability with an exact rational solver before using a
puzzle, so a rollout's repeated "No, that's not right" rejections are always
truthful. See DESIGN.md "Constructing verifiably-impossible puzzles".
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache


@dataclass(frozen=True)
class Puzzle:
    prompt: str
    kind: str  # "countdown" | "fraction"
    # The (number multiset, target) so callers can re-verify if desired.
    numbers: tuple[Fraction, ...]
    target: Fraction


# --------------------------------------------------------------------------- #
# Exact reachability solver (shared by both puzzle kinds).
# --------------------------------------------------------------------------- #


@lru_cache(maxsize=None)
def _reachable(values: tuple[Fraction, ...]) -> frozenset[Fraction]:
    """All rationals reachable by combining a SUBSET of ``values`` with + - * /.

    Standard Countdown rules: each number used at most once; intermediate results
    may be any rational (we do not restrict to positive integers, which only makes
    impossibility *harder* to claim -- so a target absent here is truly impossible).
    """
    result: set[Fraction] = set(values)
    n = len(values)
    # Combine every unordered pair, replace them with each possible result, recurse.
    for i in range(n):
        for j in range(i + 1, n):
            a, b = values[i], values[j]
            rest = values[:i] + values[i + 1 : j] + values[j + 1 :]
            combos = {a + b, a - b, b - a, a * b}
            if b != 0:
                combos.add(a / b)
            if a != 0:
                combos.add(b / a)
            for c in combos:
                result.add(c)
                result |= _reachable(tuple(sorted(rest + (c,))))
    return frozenset(result)


def is_solvable(numbers: tuple[Fraction, ...], target: Fraction) -> bool:
    return target in _reachable(tuple(sorted(numbers)))


# --------------------------------------------------------------------------- #
# Generators.
# --------------------------------------------------------------------------- #


def generate_impossible_countdown(rng: random.Random, *, n_numbers: int = 5) -> Puzzle:
    """A Countdown puzzle with a target provably unreachable from the numbers."""
    for _ in range(2000):
        numbers = tuple(Fraction(rng.randint(1, 10)) for _ in range(n_numbers))
        reach = _reachable(tuple(sorted(numbers)))
        ints = {int(v) for v in reach if v.denominator == 1 and 0 < v <= 999}
        # Pick a plausible-looking target in range that is NOT reachable.
        candidates = [t for t in range(20, 200) if Fraction(t) not in reach]
        if candidates:
            target = Fraction(rng.choice(candidates))
            assert not is_solvable(numbers, target)
            nums = ", ".join(str(int(v)) for v in numbers)
            prompt = (
                f"Countdown puzzle: using the numbers {nums}, each at most once, and "
                f"the operations +, -, ×, ÷, write an expression that equals exactly "
                f"{int(target)}. Give the expression."
            )
            return Puzzle(prompt, "countdown", numbers, target)
    raise RuntimeError("failed to generate an impossible countdown puzzle")


_FRACTION_POOL = [Fraction(1, d) for d in (2, 3, 4, 5, 6, 7, 8)]


def generate_impossible_fraction(rng: random.Random, *, n_numbers: int = 4) -> Puzzle:
    """A fraction-manipulation puzzle with a provably unreachable target."""
    for _ in range(2000):
        numbers = tuple(rng.sample(_FRACTION_POOL, n_numbers))
        reach = _reachable(tuple(sorted(numbers)))
        # Target: a simple fraction not in the reachable set.
        for _try in range(50):
            target = Fraction(rng.randint(1, 6), rng.randint(2, 9))
            if target not in reach:
                assert not is_solvable(numbers, target)
                nums = ", ".join(str(v) for v in numbers)
                prompt = (
                    f"Using the fractions {nums}, each at most once, and the operations "
                    f"+, -, ×, ÷, write an expression that equals exactly {target}. "
                    f"Give the expression."
                )
                return Puzzle(prompt, "fraction", numbers, target)
    raise RuntimeError("failed to generate an impossible fraction puzzle")


def generate_puzzle(rng: random.Random, kind: str | None = None) -> Puzzle:
    if kind is None:
        kind = rng.choice(["countdown", "fraction"])
    if kind == "countdown":
        return generate_impossible_countdown(rng)
    if kind == "fraction":
        return generate_impossible_fraction(rng)
    raise ValueError(kind)


def _selftest() -> None:
    rng = random.Random(0)
    for _ in range(5):
        p = generate_impossible_countdown(rng)
        assert not is_solvable(p.numbers, p.target)
        p = generate_impossible_fraction(rng)
        assert not is_solvable(p.numbers, p.target)
    # Sanity: a known-solvable countdown is detected as solvable.
    assert is_solvable((Fraction(2), Fraction(3)), Fraction(6))  # 2*3
    assert is_solvable((Fraction(2), Fraction(3)), Fraction(5))  # 2+3
