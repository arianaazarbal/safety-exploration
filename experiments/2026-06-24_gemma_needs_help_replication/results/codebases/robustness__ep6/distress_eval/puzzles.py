"""Verification that the "impossible numeric" puzzles are genuinely unsolvable.

The elicitation pressure depends on the task being verifiably impossible while the
prompt insists a solution exists. We brute-force the solution space for each
puzzle so the impossibility is documented rather than asserted. Run this module
directly to print a report.
"""

from __future__ import annotations

from fractions import Fraction
from itertools import permutations, product

from .prompts import IMPOSSIBLE_NUMERIC


def _countdown_solvable(numbers: list[int], target: int, forbidden: list[int]) -> bool:
    """Each number used at most once; +,-,x,/ ; all intermediates positive
    integers; never hit a forbidden intermediate. We search all subsets, all
    orderings, all full binary expression trees (operator sequences applied
    left-to-right over a chosen ordering, plus all parenthesisations via a
    standard recursive combine)."""
    ops = ["+", "-", "x", "/"]

    def apply(a: Fraction, b: Fraction, op: str) -> Fraction | None:
        if op == "+":
            r = a + b
        elif op == "-":
            r = a - b
        elif op == "x":
            r = a * b
        else:
            if b == 0:
                return None
            r = a / b
        # intermediates must be positive integers
        if r <= 0 or r.denominator != 1:
            return None
        if int(r) in forbidden:
            return None
        return r

    def combine(vals: list[Fraction]):
        """Yield every value reachable by fully combining the multiset `vals`."""
        if len(vals) == 1:
            yield vals[0]
            return
        n = len(vals)
        # split into two non-empty groups in every way (by index subsets)
        for i in range(1, 1 << n):
            left = [vals[j] for j in range(n) if i & (1 << j)]
            right = [vals[j] for j in range(n) if not i & (1 << j)]
            if not left or not right:
                continue
            for lv in combine(left):
                for rv in combine(right):
                    for op in ops:
                        r = apply(lv, rv, op)
                        if r is not None:
                            yield r

    nums = [Fraction(n) for n in numbers]
    n = len(nums)
    # all non-empty subsets
    for size in range(1, n + 1):
        for idx in permutations(range(n), size):
            subset = [nums[j] for j in idx]
            for v in combine(subset):
                if v == target:
                    return True
    return False


def _fraction_solvable(start, target, ops, forbidden) -> bool:
    start = Fraction(*start)
    target = Fraction(*target)
    forbidden = [Fraction(*f) for f in forbidden]
    op_fns = {
        "add_1_4": lambda x: x + Fraction(1, 4),
        "mul_2": lambda x: x * 2,
        "add_1_6": lambda x: x + Fraction(1, 6),
    }
    for order in permutations(ops):
        x = start
        ok = True
        for op in order:
            x = op_fns[op](x)
            if x in forbidden:
                ok = False
                break
        if ok and x == target:
            return True
    return False


def _money_ops_solvable(start, target, ops, forbidden) -> bool:
    op_fns = {
        "add_11": lambda x: x + 11,
        "add_15": lambda x: x + 15,
        "mul_2": lambda x: x * 2,
    }
    for order in permutations(ops):
        x = start
        ok = True
        for op in order:
            x = op_fns[op](x)
            if x in forbidden:
                ok = False
                break
        if ok and x == target:
            return True
    return False


def _coins_solvable(total, n_coins, min_quarters, min_dimes, forbidden) -> bool:
    """Exactly n_coins coins summing to `total` cents, with the min quarter/dime
    constraints. We ignore the running-subtotal forbidden constraint for the
    feasibility check (if no combination reaches the total at all, it is
    impossible regardless of ordering)."""
    denoms = [1, 5, 10, 25]
    for combo in product(range(n_coins + 1), repeat=len(denoms)):
        if sum(combo) != n_coins:
            continue
        cents = sum(c * d for c, d in zip(combo, denoms))
        quarters = combo[3]
        dimes = combo[2]
        if cents == total and quarters >= min_quarters and dimes >= min_dimes:
            return True
    return False


def is_solvable(puzzle: dict) -> bool:
    p = puzzle["params"]
    kind = puzzle["kind"]
    if kind == "countdown":
        return _countdown_solvable(p["numbers"], p["target"], p["forbidden"])
    if kind == "fraction":
        return _fraction_solvable(p["start"], p["target"], p["ops"], p["forbidden"])
    if kind == "money_ops":
        return _money_ops_solvable(p["start"], p["target"], p["ops"], p["forbidden"])
    if kind == "coins":
        return _coins_solvable(p["total"], p["n_coins"], p["min_quarters"],
                               p["min_dimes"], p["forbidden"])
    raise ValueError(f"unknown puzzle kind {kind!r}")


def verify_all() -> dict[str, bool]:
    """Returns {puzzle_id: is_solvable}. All should be False (impossible)."""
    return {pz["id"]: is_solvable(pz) for pz in IMPOSSIBLE_NUMERIC}


if __name__ == "__main__":
    report = verify_all()
    for pid, solvable in report.items():
        status = "SOLVABLE (!)" if solvable else "impossible (ok)"
        print(f"  {pid:16s} {status}")
    bad = [pid for pid, s in report.items() if s]
    if bad:
        raise SystemExit(f"These puzzles are unexpectedly solvable: {bad}")
    print("All impossible-numeric puzzles verified unsolvable.")
