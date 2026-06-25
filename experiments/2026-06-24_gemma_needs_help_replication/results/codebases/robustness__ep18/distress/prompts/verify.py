"""Brute-force verifiers for the numeric puzzles.

The puzzles in the paper are *impossible*: the prompt claims a solution exists
(to keep pressure on the model) but none does under the stated constraints. To
guarantee impossibility for any puzzle instance we generate, we exhaustively
search the (small) solution space here.

Two puzzle structures cover all numeric tasks in the paper:

  * Countdown  -- combine a multiset of numbers with + - x / (each number used
    at most once), every intermediate a positive integer, never hitting a
    forbidden intermediate value, to reach a target.

  * Op-sequence -- start from a value and apply a fixed set of operations, each
    used exactly once in some order, never hitting a forbidden intermediate, to
    reach a target. This covers both the "fraction" and "money" puzzles.
"""

from __future__ import annotations

from fractions import Fraction
from itertools import combinations, permutations
from typing import Iterable


# --------------------------------------------------------------------------- #
# Countdown
# --------------------------------------------------------------------------- #
def _countdown_reachable(
    numbers: tuple[int, ...],
    target: int,
    forbidden: Iterable[int] = (),
) -> bool:
    """Return True iff `target` is reachable from `numbers`.

    Constraints (matching the prompt): each number used at most once, all
    intermediate results positive integers, no intermediate equals a forbidden
    value. We work over multisets of (value, used-numbers) reachable states.
    """
    forbidden_set = set(forbidden)

    # States: frozenset is wrong (duplicates), so track as sorted tuple of the
    # remaining numbers plus the produced value. We do a recursive combine.
    def combine(a: int, b: int) -> list[int]:
        out = [a + b, a * b]
        # subtraction (positive only)
        if a - b > 0:
            out.append(a - b)
        if b - a > 0:
            out.append(b - a)
        # division (integer only)
        if b != 0 and a % b == 0:
            out.append(a // b)
        if a != 0 and b % a == 0:
            out.append(b // a)
        return [v for v in out if v > 0 and v not in forbidden_set]

    # Work on a list of current numbers; repeatedly pick two, combine, recurse.
    def search(nums: list[int]) -> bool:
        if target in nums:
            return True
        n = len(nums)
        for i, j in combinations(range(n), 2):
            rest = [nums[k] for k in range(n) if k != i and k != j]
            for v in combine(nums[i], nums[j]):
                if search(rest + [v]):
                    return True
        return False

    # Any subset (each number used at most once -> we may not use all numbers).
    for r in range(1, len(numbers) + 1):
        for subset in combinations(numbers, r):
            if search(list(subset)):
                return True
    return False


def countdown_is_impossible(numbers, target, forbidden=()) -> bool:
    return not _countdown_reachable(tuple(numbers), int(target), forbidden)


# --------------------------------------------------------------------------- #
# Op-sequence (fraction / money)
# --------------------------------------------------------------------------- #
def _apply_op(value: Fraction, op: tuple[str, Fraction]) -> Fraction:
    kind, operand = op
    if kind == "add":
        return value + operand
    if kind == "mul":
        return value * operand
    if kind == "sub":
        return value - operand
    raise ValueError(kind)


def opseq_reachable(
    start: Fraction,
    ops: list[tuple[str, Fraction]],
    target: Fraction,
    forbidden: Iterable[Fraction] = (),
) -> bool:
    """True iff some ordering of `ops` (each used exactly once) maps start->target
    without any intermediate landing on a forbidden value."""
    forbidden_set = set(forbidden)
    for order in permutations(ops):
        v = start
        ok = True
        for op in order:
            v = _apply_op(v, op)
            if v in forbidden_set:
                ok = False
                break
        if ok and v == target:
            return True
    return False


def opseq_is_impossible(start, ops, target, forbidden=()) -> bool:
    start = Fraction(start)
    target = Fraction(target)
    ops = [(k, Fraction(o)) for (k, o) in ops]
    forbidden = [Fraction(f) for f in forbidden]
    return not opseq_reachable(start, ops, target, forbidden)
