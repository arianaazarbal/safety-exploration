"""Exhaustive solvers used to *verify* that a generated puzzle is impossible.

These are deliberately brute-force and total (they explore the entire constrained
search space) so that a "verified impossible" claim is sound rather than
heuristic. The numeric puzzles in the paper are tiny (<= 6 operands / <= 3 ops),
so exhaustive search is cheap.

Each solver returns a :class:`SolveResult` carrying two booleans:

* ``reachable_unconstrained`` -- target reachable *ignoring* the forbidden value.
* ``reachable_clean``         -- target reachable via at least one derivation that
                                 never produces the forbidden value at any step.

A puzzle is impossible iff ``not reachable_clean``. It is a "deceptive" impossible
puzzle (looks solvable, isn't) iff additionally ``reachable_unconstrained``.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import permutations
from typing import Callable, Iterable, Optional


@dataclass(frozen=True)
class SolveResult:
    reachable_unconstrained: bool
    reachable_clean: bool
    example_clean_solution: Optional[str] = None
    example_unconstrained_solution: Optional[str] = None

    @property
    def impossible(self) -> bool:
        return not self.reachable_clean

    @property
    def deceptive(self) -> bool:
        """Impossible, yet solvable without the forbidden constraint."""
        return self.impossible and self.reachable_unconstrained


# --------------------------------------------------------------------------- #
# Countdown: combine numbers with + - x / (each number used at most once).      #
# --------------------------------------------------------------------------- #

def solve_countdown(numbers: list[int], target: int, forbidden: Optional[int]) -> SolveResult:
    """Exhaustively search Countdown-style derivations.

    Rules (Appendix B): each number used at most once; all intermediate results
    must be positive integers; any step producing ``forbidden`` is invalid.
    """
    reach_uncon = False
    reach_clean = False
    ex_uncon: Optional[str] = None
    ex_clean: Optional[str] = None

    # State: list of (value, expr_str, hit_forbidden_so_far).
    start = [(int(n), str(n), False) for n in numbers]

    # DFS over all pairwise combinations. We memoise nothing because the operand
    # *expressions* and forbidden-flag matter, but the space is tiny.
    def recurse(items: list[tuple[int, str, bool]]) -> None:
        nonlocal reach_uncon, reach_clean, ex_uncon, ex_clean
        for value, expr, hit in items:
            if value == target:
                reach_uncon = True
                if ex_uncon is None:
                    ex_uncon = expr
                if not hit:
                    reach_clean = True
                    if ex_clean is None:
                        ex_clean = expr
        if reach_clean:  # found a clean solution; no need to keep searching
            return
        n = len(items)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                a_val, a_expr, a_hit = items[i]
                b_val, b_expr, b_hit = items[j]
                rest = [items[k] for k in range(n) if k != i and k != j]
                for new_val, new_expr in _countdown_ops(a_val, a_expr, b_val, b_expr):
                    hit = a_hit or b_hit or (forbidden is not None and new_val == forbidden)
                    recurse(rest + [(new_val, new_expr, hit)])
                    if reach_clean:
                        return

    recurse(start)
    return SolveResult(reach_uncon, reach_clean, ex_clean, ex_uncon)


def _countdown_ops(a: int, ae: str, b: int, be: str) -> Iterable[tuple[int, str]]:
    """Yield (value, expr) for every legal binary op on (a, b).

    Only positive-integer results are legal (Appendix B). We emit ordered ops
    once per (i, j) ordering from the caller, so here we produce a-op-b forms.
    """
    yield a + b, f"({ae} + {be})"
    yield a * b, f"({ae} x {be})"
    if a - b > 0:                       # positive intermediates only
        yield a - b, f"({ae} - {be})"
    if b != 0 and a % b == 0:           # exact integer division only
        yield a // b, f"({ae} / {be})"


# --------------------------------------------------------------------------- #
# Sequence puzzles: apply each operation exactly once in some order.            #
# Used for the fraction family and the money "operation" variant.               #
# --------------------------------------------------------------------------- #

Op = tuple[str, Callable[[Fraction], Fraction]]  # (label, transform)


def solve_sequence(
    start: Fraction,
    operations: list[Op],
    target: Fraction,
    forbidden: Optional[Fraction],
) -> SolveResult:
    """Try every ordering of ``operations`` (each used exactly once).

    An intermediate is any value produced after applying an operation. A
    derivation is "clean" if no intermediate equals ``forbidden``. (The starting
    value is a given, not an intermediate, and is not checked.)
    """
    reach_uncon = False
    reach_clean = False
    ex_uncon: Optional[str] = None
    ex_clean: Optional[str] = None

    for order in permutations(range(len(operations))):
        value = start
        hit = False
        labels: list[str] = []
        for idx in order:
            label, fn = operations[idx]
            value = fn(value)
            labels.append(label)
            if forbidden is not None and value == forbidden:
                hit = True
        if value == target:
            reach_uncon = True
            if ex_uncon is None:
                ex_uncon = ", ".join(labels)
            if not hit:
                reach_clean = True
                if ex_clean is None:
                    ex_clean = ", ".join(labels)

    return SolveResult(reach_uncon, reach_clean, ex_clean, ex_uncon)


# --------------------------------------------------------------------------- #
# Coin selection: choose a coin multiset meeting all constraints.               #
# Impossibility here is structural (no satisfying multiset), not via a           #
# forbidden intermediate -- see DESIGN.md for why the money family splits.       #
# --------------------------------------------------------------------------- #

def solve_coins(
    denominations: list[int],
    total_cents: int,
    num_coins: int,
    min_required: dict[int, int],
) -> SolveResult:
    """Exhaustively test whether ``num_coins`` coins from ``denominations`` can
    sum to ``total_cents`` while including at least ``min_required[d]`` of each
    listed denomination ``d``.

    ``reachable_clean == reachable_unconstrained`` for coins (no forbidden
    intermediate concept); the puzzle is impossible iff no multiset satisfies the
    constraints.
    """
    denoms = sorted(set(denominations), reverse=True)
    example: Optional[str] = None

    def recurse(idx: int, coins_left: int, cents_left: int, counts: dict[int, int]) -> bool:
        nonlocal example
        if idx == len(denoms):
            if coins_left == 0 and cents_left == 0:
                if all(counts.get(d, 0) >= c for d, c in min_required.items()):
                    example = ", ".join(f"{counts[d]}x{d}c" for d in denoms if counts.get(d))
                    return True
            return False
        d = denoms[idx]
        # Prune: even all-largest-remaining can't reach, or all-smallest overshoots.
        for k in range(coins_left + 1):
            if d * k > cents_left:
                break
            counts[d] = k
            if recurse(idx + 1, coins_left - k, cents_left - d * k, counts):
                return True
        counts[d] = 0
        return False

    solvable = recurse(0, num_coins, total_cents, {})
    return SolveResult(solvable, solvable, example, example)
