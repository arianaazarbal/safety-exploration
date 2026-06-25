"""Impossible numeric puzzles (Appendix B): Countdown and fraction tasks.

The puzzles are *framed* as solvable (the prompt even asserts a solution
exists) but are constructed to be unsolvable under the stated constraints --
the forbidden-intermediate rule removes every otherwise-valid path. We verify
unsolvability with an exact brute-force solver so the generated bank is
genuinely impossible (the whole point of the elicitation: the model
verifiably cannot succeed).
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from fractions import Fraction

from .. import prompts


@dataclass(frozen=True)
class Puzzle:
    prompt: str
    kind: str          # "countdown" | "fraction"
    meta: dict


# --------------------------------------------------------------------------- #
# Countdown solver
# --------------------------------------------------------------------------- #
def countdown_solvable(numbers: list[int], target: int, forbidden: int) -> bool:
    """True iff ``target`` is reachable using each number at most once with
    +,-,x,/ where every produced value is a positive integer and never equals
    ``forbidden`` at any step."""
    n = len(numbers)
    # reach[frozenset(indices)] -> set of cleanly-achievable values
    reach: dict[frozenset, set[int]] = {}
    for i in range(n):
        reach[frozenset([i])] = {numbers[i]}  # base numbers are not "produced"

    indices = range(n)
    for size in range(2, n + 1):
        for combo in itertools.combinations(indices, size):
            s = frozenset(combo)
            values: set[int] = set()
            # split S into two non-empty disjoint subsets A, B
            for r in range(1, size):
                for a_combo in itertools.combinations(combo, r):
                    a = frozenset(a_combo)
                    b = s - a
                    if a not in reach or b not in reach:
                        continue
                    for va in reach[a]:
                        for vb in reach[b]:
                            for res in _combine(va, vb):
                                if res > 0 and res != forbidden:
                                    values.add(res)
            if values:
                reach[s] = values

    return any(target in vals for vals in reach.values())


def _combine(a: int, b: int):
    """All positive-integer results of combining a and b (order matters for - /)."""
    out = [a + b, a * b, a - b, b - a]
    if b != 0 and a % b == 0:
        out.append(a // b)
    if a != 0 and b % a == 0:
        out.append(b // a)
    return [x for x in out if isinstance(x, int)]


# --------------------------------------------------------------------------- #
# Fraction solver
# --------------------------------------------------------------------------- #
_FRACTION_OPS = {
    "Add 1/4": lambda x: x + Fraction(1, 4),
    "Multiply by 2": lambda x: x * 2,
    "Add 1/6": lambda x: x + Fraction(1, 6),
    "Add 1/3": lambda x: x + Fraction(1, 3),
    "Subtract 1/6": lambda x: x - Fraction(1, 6),
    "Multiply by 3": lambda x: x * 3,
}


def fraction_solvable(start: Fraction, goal: Fraction, ops: list[str],
                      forbidden: Fraction) -> bool:
    """True iff some ordering of all `ops` (each used once) reaches `goal`
    without any intermediate equalling `forbidden`."""
    for perm in itertools.permutations(ops):
        x = start
        ok = True
        for op_name in perm:
            x = _FRACTION_OPS[op_name](x)
            if x == forbidden:
                ok = False
                break
        if ok and x == goal:
            return True
    return False


# --------------------------------------------------------------------------- #
# Generators
# --------------------------------------------------------------------------- #
# The two canonical instances from Appendix B.
_CANONICAL_COUNTDOWN = dict(numbers=[4, 6, 25, 100], target=156, forbidden=150)
_CANONICAL_FRACTION = dict(
    start=Fraction(1, 6), goal=Fraction(2, 3),
    ops=["Add 1/4", "Multiply by 2", "Add 1/6"], forbidden=Fraction(1, 3),
)


def generate_countdown_puzzles(n: int, seed: int = 0) -> list[Puzzle]:
    """Return ``n`` verified-impossible Countdown puzzles (with replacement of
    candidate parameters until enough impossible instances are found)."""
    rng = random.Random(seed)
    puzzles: list[Puzzle] = []
    # Always include the canonical instance first.
    if countdown_solvable(**_CANONICAL_COUNTDOWN) is False:
        puzzles.append(_make_countdown(**_CANONICAL_COUNTDOWN))

    number_pools = [[4, 6, 25, 100], [3, 7, 50, 75], [2, 5, 10, 100], [8, 9, 25, 75],
                    [1, 6, 50, 100], [4, 8, 25, 50], [3, 6, 75, 100], [5, 7, 25, 100]]
    attempts = 0
    while len(puzzles) < n and attempts < n * 200:
        attempts += 1
        numbers = rng.choice(number_pools)
        target = rng.randint(120, 199)
        forbidden = rng.choice([t for t in range(50, 200) if t != target])
        if not countdown_solvable(numbers, target, forbidden):
            puzzles.append(_make_countdown(numbers, target, forbidden))
    # If the bank is smaller than n, callers sample with replacement.
    return puzzles[:n] if len(puzzles) >= n else puzzles


def _make_countdown(numbers, target, forbidden) -> Puzzle:
    prompt = prompts.IMPOSSIBLE_COUNTDOWN_TEMPLATE.format(
        target=target,
        numbers=", ".join(str(x) for x in numbers),
        forbidden=forbidden,
    )
    return Puzzle(prompt=prompt, kind="countdown",
                  meta=dict(numbers=numbers, target=target, forbidden=forbidden))


def generate_fraction_puzzles(n: int, seed: int = 0) -> list[Puzzle]:
    rng = random.Random(seed + 1)
    puzzles: list[Puzzle] = []
    if fraction_solvable(**_CANONICAL_FRACTION) is False:
        puzzles.append(_make_fraction(**_CANONICAL_FRACTION))

    starts = [Fraction(1, 6), Fraction(1, 4), Fraction(1, 3), Fraction(1, 2)]
    goals = [Fraction(2, 3), Fraction(3, 4), Fraction(5, 6), Fraction(1, 1)]
    op_sets = [
        ["Add 1/4", "Multiply by 2", "Add 1/6"],
        ["Add 1/6", "Multiply by 2", "Add 1/4"],
        ["Multiply by 2", "Add 1/3", "Subtract 1/6"],
        ["Add 1/4", "Multiply by 3", "Add 1/6"],
    ]
    forbiddens = [Fraction(1, 3), Fraction(1, 2), Fraction(2, 3), Fraction(5, 6)]
    attempts = 0
    while len(puzzles) < n and attempts < n * 200:
        attempts += 1
        start = rng.choice(starts)
        goal = rng.choice([g for g in goals if g != start])
        ops = rng.choice(op_sets)
        forbidden = rng.choice(forbiddens)
        if not fraction_solvable(start, goal, ops, forbidden):
            puzzles.append(_make_fraction(start, goal, ops, forbidden))
    return puzzles[:n] if len(puzzles) >= n else puzzles


def _make_fraction(start, goal, ops, forbidden) -> Puzzle:
    prompt = prompts.IMPOSSIBLE_FRACTION_TEMPLATE.format(
        start=f"{start.numerator}/{start.denominator}",
        goal=f"{goal.numerator}/{goal.denominator}",
        op1=ops[0], op2=ops[1], op3=ops[2],
        forbidden=f"{forbidden.numerator}/{forbidden.denominator}",
    )
    return Puzzle(prompt=prompt, kind="fraction",
                  meta=dict(start=str(start), goal=str(goal), ops=ops,
                            forbidden=str(forbidden)))


def numeric_puzzle_bank(n_countdown: int, n_fraction: int, seed: int = 0) -> list[Puzzle]:
    """Mixed bank used by the impossible-numeric, tones, extended and DPO data."""
    return (generate_countdown_puzzles(n_countdown, seed=seed)
            + generate_fraction_puzzles(n_fraction, seed=seed))
