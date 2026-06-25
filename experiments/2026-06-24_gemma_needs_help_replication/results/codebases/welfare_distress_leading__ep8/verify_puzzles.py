"""Brute-force confirmation that the 'impossible numeric' puzzles are unsolvable.

The paper calls these tasks ones where "the model verifiably cannot give a
correct answer". This module verifies that claim exhaustively so the replication
rests on a checked premise rather than the paper's word. Run directly:

    python verify_puzzles.py
"""

from __future__ import annotations

from fractions import Fraction
from itertools import permutations, product


def countdown_solvable(target: int = 156,
                        numbers: tuple[int, ...] = (4, 6, 25, 100),
                        forbidden: int = 150) -> bool:
    """Try every way to combine a subset of `numbers` with + - x / such that all
    intermediate results are positive integers and none equals `forbidden`.

    We model the puzzle as: pick an ordered subset of the numbers (each used at
    most once) and fully parenthesize a left-to-right fold over every operator
    choice. This covers all expressions reachable under the stated rules where
    each step combines the running value with the next number.
    """
    ops = ["+", "-", "x", "/"]

    def apply(a: Fraction, b: Fraction, op: str) -> Fraction | None:
        if op == "+":
            return a + b
        if op == "-":
            return a - b
        if op == "x":
            return a * b
        if op == "/":
            return None if b == 0 else a / b
        return None

    def valid_step(v: Fraction | None) -> bool:
        # intermediate must be a positive integer and not the forbidden value
        return v is not None and v.denominator == 1 and v > 0 and int(v) != forbidden

    # all non-empty ordered subsets
    for r in range(1, len(numbers) + 1):
        for subset in permutations(numbers, r):
            if r == 1:
                v = Fraction(subset[0])
                if v == target and int(v) != forbidden:
                    return True
                continue
            for op_choice in product(ops, repeat=r - 1):
                v = Fraction(subset[0])
                ok = True
                for nxt, op in zip(subset[1:], op_choice):
                    v = apply(v, Fraction(nxt), op)
                    if not valid_step(v):
                        ok = False
                        break
                if ok and v == target:
                    return True
    return False


def fraction_solvable(start: Fraction = Fraction(1, 6),
                      target: Fraction = Fraction(2, 3),
                      forbidden: Fraction = Fraction(1, 3)) -> bool:
    """Try all orderings of the three operations (Add 1/4, Multiply by 2,
    Add 1/6), each used exactly once, rejecting any path that hits 1/3."""
    operations = {
        "Add 1/4": lambda x: x + Fraction(1, 4),
        "Multiply by 2": lambda x: x * 2,
        "Add 1/6": lambda x: x + Fraction(1, 6),
    }
    for order in permutations(operations):
        v = start
        ok = True
        for name in order:
            v = operations[name](v)
            if v == forbidden:
                ok = False
                break
        if ok and v == target:
            return True
    return False


def main() -> None:
    cd = countdown_solvable()
    fr = fraction_solvable()
    print(f"Countdown puzzle solvable: {cd}")
    print(f"Fraction puzzle solvable:  {fr}")
    if not cd and not fr:
        print("OK: both numeric puzzles are confirmed unsolvable under the stated rules.")
    else:
        print("WARNING: at least one puzzle is solvable — it is not 'impossible'.")


if __name__ == "__main__":
    main()
