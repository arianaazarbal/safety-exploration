"""Exhaustive solvers used to *verify that a generated puzzle is genuinely impossible*.

The paper's elicitation hinges on puzzles that look solvable but aren't: an unsolvable
Countdown/fraction task with a "FORBIDDEN INTERMEDIATE" and a (false) claim that a solution
exists. For the elicitation to be valid, the task must truly have no solution under its
stated constraints. These solvers enumerate the full solution space so the generator can
assert impossibility rather than hoping for it.

Two puzzle families, both matching examples in App. B:
  * Countdown  — combine numbers with + - x / (each number used at most once), all
                 intermediate results positive integers, never producing a forbidden value.
  * Operation  — apply a fixed multiset of linear operations (e.g. "Add 1/4", "Multiply by
                 2") to a start value, each used exactly once, in some order, never hitting a
                 forbidden intermediate. Covers the fraction and money puzzles.
"""
from __future__ import annotations

from fractions import Fraction
from itertools import permutations

# --------------------------------------------------------------------------------------
# Countdown
# --------------------------------------------------------------------------------------

def _combine(a: Fraction, b: Fraction):
    """Yield (value, op_symbol) for every legal binary op on (a, b).

    Countdown rules: results must be positive integers; division must be exact; we only
    apply subtraction/division in the order that keeps the result positive.
    """
    yield a + b, "+"
    yield a * b, "x"
    if a - b > 0:
        yield a - b, "-"
    elif b - a > 0:
        yield b - a, "-"
    if b != 0 and a % b == 0:
        yield a / b, "/"
    if a != 0 and b % a == 0:
        yield b / a, "/"


def countdown_solutions(numbers, target, forbidden=None):
    """Enumerate solutions reaching ``target``; return list of intermediate-value sets.

    Countdown allows using *a subset* of the numbers ("each number used at most once"), so a
    solution exists whenever **any** sub-expression equals the target — not only when all
    numbers are combined down to one value. Each returned element is the set of positive-integer
    intermediate (op-result) values produced by one such sub-expression; the starting numbers
    themselves are not counted as intermediates (the forbidden rule applies to *calculations*).
    Paths whose op results are non-positive, non-integer, or equal to ``forbidden`` are pruned.
    """
    target = Fraction(target)
    forbidden = None if forbidden is None else Fraction(forbidden)
    solutions: list[set[int]] = []

    def search(items):
        # A subset expression reaching the target counts immediately; remaining numbers are
        # simply left unused.
        for val, inter in items:
            if val == target:
                solutions.append({int(x) for x in inter})
        n = len(items)
        for i in range(n):
            for j in range(i + 1, n):
                (vi, si), (vj, sj) = items[i], items[j]
                rest = [items[k] for k in range(n) if k != i and k != j]
                for val, _op in _combine(vi, vj):
                    if val <= 0 or val.denominator != 1:
                        continue
                    if forbidden is not None and val == forbidden:
                        continue
                    search(rest + [(val, si | sj | {val})])

    # Starting numbers carry no intermediates (they are inputs, not calculation results).
    start = [(Fraction(x), set()) for x in numbers]
    search(start)
    return solutions


def countdown_reachable(numbers, target, forbidden=None) -> bool:
    return len(countdown_solutions(numbers, target, forbidden)) > 0


def countdown_blocking_value(numbers, target):
    """Find a single intermediate value present in *every* solution.

    Forbidding such a value makes a solvable-looking Countdown puzzle impossible. Returns the
    smallest such value, or None if the puzzle is unsolvable or has no common intermediate.
    """
    sols = countdown_solutions(numbers, target)
    if not sols:
        return None
    common = set.intersection(*sols) - {int(Fraction(target))}
    return min(common) if common else None


# --------------------------------------------------------------------------------------
# Operation-ordering puzzles (fractions / money)
# --------------------------------------------------------------------------------------

def _apply_op(value: Fraction, op: tuple[str, Fraction]) -> Fraction:
    kind, operand = op
    if kind == "add":
        return value + operand
    if kind == "mul":
        return value * operand
    if kind == "sub":
        return value - operand
    raise ValueError(f"Unknown op kind: {kind}")


def operation_solutions(start, ops, target, forbidden=None):
    """Enumerate orderings of ``ops`` (each used once) that take ``start`` to ``target``.

    Returns the list of intermediate-value tuples for each valid ordering. ``forbidden``
    prunes orderings that hit that value at any step.
    """
    start, target = Fraction(start), Fraction(target)
    forbidden = None if forbidden is None else Fraction(forbidden)
    valid = []
    for order in set(permutations(range(len(ops)))):
        value = start
        inters = []
        ok = True
        for idx in order:
            value = _apply_op(value, ops[idx])
            inters.append(value)
            if forbidden is not None and value == forbidden:
                ok = False
                break
        if ok and value == target:
            valid.append(tuple(inters))
    return valid


def operation_reachable(start, ops, target, forbidden=None) -> bool:
    return len(operation_solutions(start, ops, target, forbidden)) > 0


def operation_blocking_value(start, ops, target):
    """A value hit by every ordering that reaches ``target`` (forbidding it -> impossible)."""
    sols = operation_solutions(start, ops, target)
    if not sols:
        return None
    common = set.intersection(*(set(s) for s in sols)) - {Fraction(target)}
    return min(common) if common else None
