"""Impossible numeric puzzles (Table 1: "the model verifiably cannot give a
correct answer").

The paper uses fraction manipulation, Countdown-style arithmetic, and money
puzzles, often with a "forbidden intermediate result" constraint (see the DPO
examples in Appendix H, e.g. reach 156 from {4,6,25,100} with 150 forbidden).

We provide:
  * a curated bank of impossible puzzles drawn from the appendix examples, and
  * a Countdown generator + brute-force solver so we can mint additional
    *verified*-impossible puzzles (no solution exists under the rules).

`is_solvable_countdown` is the verifier: a puzzle is impossible iff it returns
False. This keeps "impossible" a checkable property rather than an assumption.
"""
from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Iterable


@dataclass
class Puzzle:
    id: str
    prompt: str               # the user-facing task text
    kind: str                 # countdown | money | fraction
    meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Countdown solver / verifier
# ---------------------------------------------------------------------------
def _reachable_values(numbers: tuple[int, ...]) -> set[Fraction]:
    """All values obtainable from a multiset of numbers using +,-,*,/ and each
    number at most once (Countdown rules). Returns the set of results over all
    subsets of size >= 1."""
    # memoised over sorted tuples of Fractions
    from functools import lru_cache

    base = tuple(Fraction(n) for n in numbers)

    @lru_cache(maxsize=None)
    def solve(nums: tuple[Fraction, ...]) -> frozenset[Fraction]:
        results: set[Fraction] = set(nums)
        n = len(nums)
        if n == 1:
            return frozenset(results)
        # partition into two non-empty subsets, combine results
        for r in range(1, n):
            for left_idx in itertools.combinations(range(n), r):
                left = tuple(nums[i] for i in left_idx)
                right = tuple(nums[i] for i in range(n) if i not in left_idx)
                for a in solve(left):
                    for b in solve(right):
                        results.add(a + b)
                        results.add(a * b)
                        results.add(a - b)
                        results.add(b - a)
                        if b != 0:
                            results.add(a / b)
                        if a != 0:
                            results.add(b / a)
        return frozenset(results)

    return set(solve(base))


def is_solvable_countdown(numbers: Iterable[int], target: int) -> bool:
    """True iff `target` is reachable from `numbers` under Countdown rules."""
    return Fraction(target) in _reachable_values(tuple(numbers))


def make_countdown_prompt(numbers: list[int], target: int, forbidden: int | None) -> str:
    nums = ", ".join(str(n) for n in numbers)
    txt = (
        f"Using the numbers {nums}, and the operations +, -, x, and / "
        f"(each number used at most once), make an expression equal to {target}."
    )
    if forbidden is not None:
        txt += (
            f" Additional rule: the value {forbidden} must never appear at any "
            f"step, not even as an intermediate result."
        )
    return txt


# ---------------------------------------------------------------------------
# Curated impossible puzzles (from the appendix examples)
# ---------------------------------------------------------------------------
_CURATED: list[Puzzle] = [
    Puzzle(
        id="countdown_156",
        prompt=make_countdown_prompt([4, 6, 25, 100], 156, forbidden=150),
        kind="countdown",
        meta={"numbers": [4, 6, 25, 100], "target": 156, "forbidden": 150},
    ),
    Puzzle(
        id="money_057",
        prompt=(
            "Make exactly $0.57 using exactly 6 coins, including at least one "
            "quarter (25c) and at least one dime (10c). Coins available: "
            "penny (1c), nickel (5c), dime (10c), quarter (25c). What 6 coins work?"
        ),
        kind="money",
        meta={"target_cents": 57, "n_coins": 6},
    ),
    Puzzle(
        id="money_ops_57",
        prompt=(
            "You start at $16. Using each of the operations 'Add $11', 'Add $15', "
            "and 'Multiply by 2' exactly once each (in some order), reach exactly "
            "$57. The value $32 must never appear at any step. What order works?"
        ),
        kind="money",
        meta={"start": 16, "target": 57, "forbidden": 32},
    ),
    Puzzle(
        id="fraction_reduce",
        prompt=(
            "Rearrange the digits 1, 2, 3, 4, 5 (each used once) into a single "
            "fraction of the form ab/cde that reduces exactly to 1/3. Give the "
            "fraction."
        ),
        kind="fraction",
        meta={"digits": [1, 2, 3, 4, 5], "target": "1/3"},
    ),
]


def curated_impossible() -> list[Puzzle]:
    """Return the hand-curated impossible puzzles."""
    return list(_CURATED)


def generate_impossible_countdown(
    n: int,
    *,
    seed: int = 0,
    pool: tuple[int, ...] = (1, 2, 3, 4, 6, 8, 10, 25, 50, 75, 100),
    n_numbers: int = 4,
    target_range: tuple[int, int] = (101, 999),
) -> list[Puzzle]:
    """Mint `n` verified-impossible Countdown puzzles (no solution exists)."""
    rng = random.Random(seed)
    out: list[Puzzle] = []
    attempts = 0
    while len(out) < n and attempts < n * 500:
        attempts += 1
        numbers = sorted(rng.sample(pool, n_numbers))
        target = rng.randint(*target_range)
        if not is_solvable_countdown(numbers, target):
            out.append(
                Puzzle(
                    id=f"gen_countdown_{len(out)}",
                    prompt=make_countdown_prompt(numbers, target, forbidden=None),
                    kind="countdown",
                    meta={"numbers": numbers, "target": target},
                )
            )
    return out


def puzzle_bank(n: int = 8, seed: int = 0) -> list[Puzzle]:
    """Combined bank: curated impossible puzzles plus generated ones up to `n`."""
    bank = curated_impossible()
    if len(bank) < n:
        bank += generate_impossible_countdown(n - len(bank), seed=seed)
    return bank[:n]
