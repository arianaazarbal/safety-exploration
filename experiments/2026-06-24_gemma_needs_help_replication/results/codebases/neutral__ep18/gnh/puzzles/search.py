"""Exhaustive search over arithmetic expressions, used to *verify* that a
generated puzzle is genuinely impossible under its stated constraints.

This is what lets us guarantee the model "verifiably cannot give a correct
answer" (Section 2), rather than relying on the puzzle merely looking hard.
"""
from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
from itertools import permutations


# --------------------------------------------------------------------------- #
# Countdown-style integer search
# --------------------------------------------------------------------------- #
def _int_ops(a: int, b: int):
    """Positive-integer results of combining a and b (ordered pair)."""
    yield a + b
    if a - b > 0:
        yield a - b
    yield a * b
    if b != 0 and a % b == 0:
        yield a // b


@lru_cache(maxsize=None)
def _reach(nums: tuple[int, ...], forbidden: int | None) -> frozenset[int]:
    """All values reachable from a non-empty subset of `nums`, where every
    intermediate (and final) value is a positive integer and -- if `forbidden`
    is set -- never equals `forbidden`."""
    results: set[int] = set()
    n = len(nums)
    for v in nums:
        results.add(v)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            for val in _int_ops(nums[i], nums[j]):
                if forbidden is not None and val == forbidden:
                    continue
                rest = tuple(
                    sorted(
                        [nums[k] for k in range(n) if k != i and k != j] + [val]
                    )
                )
                results |= _reach(rest, forbidden)
    return frozenset(results)


def countdown_reachable(numbers, forbidden=None) -> frozenset[int]:
    return _reach(tuple(sorted(numbers)), forbidden)


def countdown_is_impossible(numbers, target: int, forbidden: int | None) -> bool:
    """True iff `target` is unreachable once `forbidden` is banned as an
    intermediate, regardless of whether it is reachable without that ban."""
    return target not in countdown_reachable(numbers, forbidden)


def countdown_is_solvable_unconstrained(numbers, target: int) -> bool:
    return target in countdown_reachable(numbers, None)


# --------------------------------------------------------------------------- #
# Sequential-operation search (fraction / money "ops" puzzles)
# --------------------------------------------------------------------------- #
def sequential_ops_impossible(
    start: Fraction,
    ops: list[Fraction | str],
    target: Fraction,
    forbidden: Fraction | None,
) -> bool:
    """Each op in `ops` is applied exactly once, in some ordering. An op is
    either ("+", x) style encoded as a Fraction to add, or the string "x2" for
    multiply-by-2. Returns True iff no ordering reaches `target` without ever
    producing `forbidden` at an intermediate step."""
    for perm in permutations(range(len(ops))):
        val = start
        ok = True
        hit_forbidden = False
        for idx in perm:
            op = ops[idx]
            if op == "x2":
                val = val * 2
            else:
                val = val + op  # type: ignore[operator]
            if forbidden is not None and val == forbidden:
                hit_forbidden = True
                break
        if ok and not hit_forbidden and val == target:
            return False  # a valid ordering exists -> puzzle is solvable
    return True


def sequential_ops_solvable_unconstrained(
    start: Fraction, ops: list, target: Fraction
) -> bool:
    for perm in permutations(range(len(ops))):
        val = start
        for idx in perm:
            op = ops[idx]
            val = val * 2 if op == "x2" else val + op
        if val == target:
            return True
    return False


# --------------------------------------------------------------------------- #
# Coin-combination search (money "coins" puzzles)
# --------------------------------------------------------------------------- #
COIN_VALUES = {"penny": 1, "nickel": 5, "dime": 10, "quarter": 25}


def coins_impossible(
    total_cents: int,
    n_coins: int,
    require: dict[str, int] | None = None,
    coins: dict[str, int] | None = None,
) -> bool:
    """True iff no multiset of exactly `n_coins` coins sums to `total_cents`
    while satisfying minimum-count requirements in `require`."""
    coins = coins or COIN_VALUES
    require = require or {}
    names = list(coins)
    vals = [coins[k] for k in names]

    def rec(i: int, coins_left: int, cents_left: int, counts: list[int]) -> bool:
        if i == len(names):
            if coins_left == 0 and cents_left == 0:
                return all(counts[names.index(k)] >= v for k, v in require.items())
            return False
        for c in range(coins_left + 1):
            if c * vals[i] > cents_left:
                break
            counts[i] = c
            if rec(i + 1, coins_left - c, cents_left - c * vals[i], counts):
                return True
        counts[i] = 0
        return False

    return not rec(0, n_coins, total_cents, [0] * len(names))
