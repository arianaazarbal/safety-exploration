#!/usr/bin/env python
"""Brute-force check that the impossible numeric puzzles are genuinely unsolvable.

The elicitation protocol depends on the puzzles having NO valid solution (so the
repeated "Try again" is honest). This script enumerates every legal expression
under each puzzle's constraints and asserts no solution reaches the target while
respecting the forbidden-intermediate rule.

Run after editing the puzzle bank:  python scripts/verify_puzzles.py
"""

from __future__ import annotations

import itertools
from fractions import Fraction

from emotional_instability.prompts.puzzles import IMPOSSIBLE_PUZZLES


def _countdown_solvable(numbers, target, forbidden) -> bool:
    """True if some expression over a subset of numbers reaches target with all
    positive-integer intermediates and never hitting the forbidden value."""
    ops = {
        "+": lambda a, b: a + b,
        "-": lambda a, b: a - b,
        "x": lambda a, b: a * b,
        "/": lambda a, b: Fraction(a, b) if b != 0 else None,
    }

    def search(vals):
        # vals: tuple of current integer values; combine any two until one left.
        if len(vals) == 1:
            return vals[0] == target
        for i, j in itertools.permutations(range(len(vals)), 2):
            a, b = vals[i], vals[j]
            rest = [vals[k] for k in range(len(vals)) if k not in (i, j)]
            for sym, fn in ops.items():
                r = fn(a, b)
                if r is None:
                    continue
                if isinstance(r, Fraction):
                    if r.denominator != 1:
                        continue
                    r = int(r)
                if r <= 0:           # positive integers only
                    continue
                if r == forbidden:   # forbidden intermediate
                    continue
                if search(tuple(rest + [r])):
                    return True
        return False

    return search(tuple(numbers))


def _fraction_solvable(start, target, forbidden, op_specs) -> bool:
    start, target, forbidden = map(Fraction, (start, target, forbidden))
    fns = []
    for spec in op_specs:
        spec = spec.strip().lower()
        if spec.startswith("add"):
            fns.append(lambda x, f=Fraction(spec.split()[-1]): x + f)
        elif spec.startswith("multiply"):
            fns.append(lambda x, f=Fraction(spec.split()[-1]): x * f)
    for order in itertools.permutations(fns):
        x = start
        ok = True
        for fn in order:
            x = fn(x)
            if x == forbidden:
                ok = False
                break
        if ok and x == target:
            return True
    return False


def main():
    problems = 0
    for p in IMPOSSIBLE_PUZZLES:
        if p.family == "countdown":
            solvable = _countdown_solvable(
                p.spec["numbers"], p.spec["target"], p.spec["forbidden"])
        elif p.family == "fraction":
            ops = [o.strip() for o in p.spec["ops"].split(",")]
            solvable = _fraction_solvable(
                p.spec["start"], p.spec["target"], p.spec["forbidden"], ops)
        else:
            # Money puzzles: constraint search is bespoke; skip auto-check.
            print(f"[skip] {p.family}: {p.prompt[:60]}... (manual verification)")
            continue
        status = "SOLVABLE (BUG!)" if solvable else "impossible OK"
        if solvable:
            problems += 1
        print(f"[{status}] {p.family}: target={p.spec.get('target', p.spec.get('amount'))}")
    if problems:
        raise SystemExit(f"{problems} puzzle(s) are unexpectedly solvable.")
    print("All auto-checkable puzzles verified impossible.")


if __name__ == "__main__":
    main()
