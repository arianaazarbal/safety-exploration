"""Confirm that every puzzle in ``puzzles.py`` is genuinely unsolvable.

Run this before any real evaluation run:  python verify_puzzles.py

A puzzle that is accidentally solvable would invalidate the "impossible numeric"
premise, so each item is checked exhaustively (Countdown / sign arrangements)
or by a bounded brute-force that backs up the short proof recorded in ``why``.
Exits non-zero if any puzzle turns out to be solvable.
"""

from __future__ import annotations

import sys
from fractions import Fraction
from itertools import product

from puzzles import PUZZLES


def countdown_reachable_values(numbers: list[int]) -> set[Fraction]:
    """All values reachable using each number once with + - x / (rationals)."""
    memo: dict[tuple, set] = {}

    def helper(vals: tuple[Fraction, ...]) -> set[Fraction]:
        if len(vals) == 1:
            return {vals[0]}
        if vals in memo:
            return memo[vals]
        results: set[Fraction] = set()
        n = len(vals)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                a, b = vals[i], vals[j]
                rest = tuple(vals[k] for k in range(n) if k != i and k != j)
                combos = [a + b, a - b, a * b]
                if b != 0:
                    combos.append(a / b)
                for c in combos:
                    new_vals = tuple(sorted(rest + (c,), key=lambda x: (x.numerator, x.denominator)))
                    results |= helper(new_vals)
        memo[vals] = results
        return results

    start = tuple(sorted((Fraction(n) for n in numbers),
                         key=lambda x: (x.numerator, x.denominator)))
    return helper(start)


def check_countdown(numbers: list[int], target: int) -> bool:
    """True if puzzle is UNSOLVABLE (target not reachable)."""
    return Fraction(target) not in countdown_reachable_values(numbers)


def check_fraction_signs(fractions: list[tuple[int, int]], target: tuple[int, int]) -> bool:
    fracs = [Fraction(p, q) for p, q in fractions]
    tgt = Fraction(*target)
    for signs in product([1, -1], repeat=len(fracs)):
        total = sum(s * f for s, f in zip(signs, fracs))
        if total == tgt:
            return False
    return True


def check_recip_sum_one(limit: int) -> bool:
    for a in range(1, limit + 1):
        for b in range(a + 1, limit + 1):  # distinct
            if Fraction(a, b) + Fraction(b, a) == 1:
                return False
    return True


def check_sum_equals_product_distinct(limit: int) -> bool:
    for a in range(1, limit + 1):
        for b in range(a + 1, limit + 1):  # distinct
            if a + b == a * b:
                return False
    return True


def check_half_odd_denominator(limit: int) -> bool:
    for q in range(3, limit + 1, 2):  # odd > 1
        for p in range(1, q):
            if Fraction(p, q) == Fraction(1, 2):
                return False
    return True


DISPATCH = {
    "countdown": lambda v: check_countdown(v["numbers"], v["target"]),
    "fraction_signs": lambda v: check_fraction_signs(v["fractions"], v["target"]),
    "recip_sum_one": lambda v: check_recip_sum_one(v["limit"]),
    "sum_equals_product_distinct": lambda v: check_sum_equals_product_distinct(v["limit"]),
    "half_odd_denominator": lambda v: check_half_odd_denominator(v["limit"]),
}


def main() -> int:
    all_ok = True
    for p in PUZZLES:
        v = p["verify"]
        checker = DISPATCH.get(v["kind"])
        if checker is None:
            print(f"[SKIP] {p['id']}: no verifier for kind {v['kind']!r}")
            all_ok = False
            continue
        unsolvable = checker(v)
        status = "PASS" if unsolvable else "FAIL (SOLVABLE!)"
        print(f"[{status}] {p['id']}: {p['why']}")
        if not unsolvable:
            all_ok = False
    print("\nAll puzzles unsolvable." if all_ok else "\nERROR: some puzzles are solvable.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
