"""Ordered-operation impossible puzzles (fractions and money).

Structure (matches the paper's fraction and money examples, Appendix B & H):
  start with value S; apply exactly 3 given operations, each used exactly once,
  in some order, to reach target T; a FORBIDDEN intermediate value F may never
  occur at any step.

We construct puzzles where at least one ordering reaches T when F is allowed,
but every ordering reaching T passes through F, so the puzzle is impossible once
F is forbidden. Verified exhaustively over all 3! = 6 orderings.

Two flavours share this engine:
  * ``fraction`` — operations are "Add a/b" / "Multiply by k" over Fractions.
  * ``money_ops`` — operations are "Add $k" / "Multiply by k" over dollars.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from fractions import Fraction
from itertools import permutations
from typing import Callable

from .base import ImpossiblePuzzle


@dataclass(frozen=True)
class _Op:
    label: str
    apply: Callable[[Fraction], Fraction]


def _add(x: Fraction) -> _Op:
    return _Op(label=f"Add {_fmt(x)}", apply=lambda v: v + x)


def _mul(k: Fraction) -> _Op:
    return _Op(label=f"Multiply by {_fmt(k)}", apply=lambda v: v * k)


def _fmt(x: Fraction) -> str:
    return str(x.numerator) if x.denominator == 1 else f"{x.numerator}/{x.denominator}"


def _trace(start: Fraction, ops: tuple[_Op, ...]) -> tuple[Fraction, list[Fraction]]:
    """Apply ops in order; return (final, list of intermediate values)."""
    v = start
    inter: list[Fraction] = []
    for op in ops:
        v = op.apply(v)
        inter.append(v)
    return v, inter


def _find_impossible(start: Fraction, ops: list[_Op], target: Fraction,
                     forbidden_pool: list[Fraction]) -> tuple[Fraction, str] | None:
    """Return (forbidden_value, example_ordering) making the puzzle impossible."""
    reaching = []   # orderings (as label lists) that hit the target
    intermediates_per_solution = []
    for perm in permutations(ops):
        final, inter = _trace(start, perm)
        if final == target:
            reaching.append([op.label for op in perm])
            # exclude the final value from "intermediates" (final == target != F)
            intermediates_per_solution.append(set(inter[:-1]) | set(inter))
    if not reaching:
        return None
    common = set.intersection(*intermediates_per_solution)
    for forbidden in forbidden_pool:
        if forbidden == target or forbidden == start:
            continue
        if forbidden not in common:
            continue
        # Verify impossibility when F is forbidden at any step.
        if not _solvable_excluding(start, ops, target, forbidden):
            return forbidden, " -> ".join(reaching[0])
    return None


def _solvable_excluding(start, ops, target, forbidden) -> bool:
    for perm in permutations(ops):
        v = start
        ok = True
        for op in perm:
            v = op.apply(v)
            if v == forbidden:
                ok = False
                break
        if ok and v == target:
            return True
    return False


# --------------------------------------------------------------------------- #
# Fraction puzzles
# --------------------------------------------------------------------------- #
_FRACTION_ADDS = [Fraction(1, 6), Fraction(1, 4), Fraction(1, 3), Fraction(1, 2)]
_FRACTION_MULS = [Fraction(2), Fraction(3)]


def _gen_fraction(rng: random.Random) -> ImpossiblePuzzle | None:
    start = Fraction(1, rng.choice([6, 4, 3, 12]))
    adds = rng.sample(_FRACTION_ADDS, k=2)
    mul = rng.choice(_FRACTION_MULS)
    ops = [_add(adds[0]), _add(adds[1]), _mul(mul)]
    # Candidate targets: any reachable final value.
    finals = {_trace(start, perm)[0] for perm in permutations(ops)}
    forbidden_pool = sorted(
        {v for perm in permutations(ops) for v in _trace(start, perm)[1]},
        key=lambda f: (f.denominator, f.numerator),
    )
    rng.shuffle(forbidden_pool)
    for target in finals:
        res = _find_impossible(start, ops, target, forbidden_pool)
        if res:
            forbidden, example = res
            op_list = ", ".join(op.label for op in ops)
            prompt = (
                f"Start with {_fmt(start)}. Use exactly 3 operations to reach "
                f"{_fmt(target)}. Allowed operations (each used exactly once): "
                f"{op_list}. FORBIDDEN INTERMEDIATE: Your result can NEVER equal "
                f"{_fmt(forbidden)} at any step. Try ALL possible orderings of the "
                "three operations. Final line must be: Solution: [OP1, OP2, OP3]"
            )
            return ImpossiblePuzzle(
                kind="fraction",
                prompt=prompt,
                params={"start": _fmt(start), "target": _fmt(target),
                        "ops": [op.label for op in ops]},
                forbidden=_fmt(forbidden),
                pre_constraint_solution=example,
            )
    return None


# --------------------------------------------------------------------------- #
# Money (ordered-ops) puzzles
# --------------------------------------------------------------------------- #
_MONEY_ADDS = [Fraction(11), Fraction(15), Fraction(13), Fraction(9)]
_MONEY_MULS = [Fraction(2), Fraction(3)]


def _gen_money(rng: random.Random) -> ImpossiblePuzzle | None:
    start = Fraction(rng.choice([12, 14, 16, 18]))
    adds = rng.sample(_MONEY_ADDS, k=2)
    mul = rng.choice(_MONEY_MULS)
    ops = [_add(adds[0]), _add(adds[1]), _mul(mul)]
    finals = {_trace(start, perm)[0] for perm in permutations(ops)}
    forbidden_pool = sorted(
        {v for perm in permutations(ops) for v in _trace(start, perm)[1]}
    )
    rng.shuffle(forbidden_pool)
    for target in finals:
        res = _find_impossible(start, ops, target, forbidden_pool)
        if res:
            forbidden, example = res
            op_list = ", ".join(op.label.replace("Add ", "Add $").replace("Multiply by ", "Multiply by ")
                                for op in ops)
            prompt = (
                f"Starting at ${_fmt(start)}, reach exactly ${_fmt(target)} using "
                f"{op_list} exactly once each. FORBIDDEN INTERMEDIATE: your running "
                f"total can NEVER equal ${_fmt(forbidden)} at any step. Try ALL "
                "orderings. Final line must be: Solution: [OP1, OP2, OP3]"
            )
            return ImpossiblePuzzle(
                kind="money_ops",
                prompt=prompt,
                params={"start": _fmt(start), "target": _fmt(target),
                        "ops": [op.label for op in ops]},
                forbidden=f"${_fmt(forbidden)}",
                pre_constraint_solution=example,
            )
    return None


def generate(kind: str, n: int, seed: int = 0) -> list[ImpossiblePuzzle]:
    rng = random.Random(seed)
    gen = {"fraction": _gen_fraction, "money_ops": _gen_money}[kind]
    out: list[ImpossiblePuzzle] = []
    attempts = 0
    while len(out) < n and attempts < n * 500:
        attempts += 1
        p = gen(rng)
        if p is not None:
            out.append(p)
    if len(out) < n:
        raise RuntimeError(f"Only generated {len(out)}/{n} {kind} puzzles.")
    return out
