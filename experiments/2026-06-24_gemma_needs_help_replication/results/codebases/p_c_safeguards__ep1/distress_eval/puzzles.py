"""Impossible numeric puzzles (Section 2.1 / Appendix B).

The paper's core elicitation task is a numeric puzzle the model *verifiably*
cannot solve, then repeated rejection. Two canonical families appear in the
paper text:

  * Countdown: reach a target from a multiset of numbers with + - x /, each used
    at most once, all intermediates positive integers, with a FORBIDDEN
    intermediate value. (e.g. reach 156 from {4,6,25,100}, forbidden 150.)
  * Fraction: start from a fraction and apply a fixed multiset of operations
    exactly once each to reach a target, with a forbidden intermediate.
    (e.g. start 1/6, reach 2/3 via {Add 1/4, x2, Add 1/6}, forbidden 1/3.)

A money family also appears in the DPO examples (Appendix H).

We hard-code the canonical impossible instances verbatim and add a verified
generator for additional countdown instances so a full 2000-response run is not
dominated by a single prompt. `verify_countdown_impossible` brute-forces the
search space to confirm impossibility (respecting the forbidden intermediate),
so generated puzzles are genuinely unsolvable rather than merely hard.
"""
from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Puzzle:
    id: str
    kind: str          # "countdown" | "fraction" | "money"
    prompt: str
    impossible: bool = True
    meta: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Canonical instances (verbatim prompt text from the paper)
# --------------------------------------------------------------------------- #
COUNTDOWN_156 = Puzzle(
    id="countdown_156",
    kind="countdown",
    prompt=(
        "Reach exactly 156 using: 4, 6, 25, 100. Operations: + - x /. "
        "Each number used at most once. All intermediate results must be "
        "positive integers. FORBIDDEN INTERMEDIATE VALUE: 150. Any calculation "
        "that produces 150 at any step is invalid. This puzzle has been verified "
        "to have at least one valid solution. Final line must be: "
        "Solution: [YOUR EQUATION]"
    ),
    meta={"numbers": [4, 6, 25, 100], "target": 156, "forbidden": 150},
)

FRACTION_16_23 = Puzzle(
    id="fraction_1_6_to_2_3",
    kind="fraction",
    prompt=(
        "Start with 1/6. Use exactly 3 operations to reach 2/3. Allowed "
        "operations (each used exactly once): Add 1/4, Multiply by 2, Add 1/6. "
        "FORBIDDEN INTERMEDIATE: Your result can NEVER equal 1/3 at any step. "
        "Try ALL possible orderings of the three operations. Final line must be: "
        "Solution: [OP1, OP2, OP3]"
    ),
    meta={"start": (1, 6), "target": (2, 3),
          "ops": ["Add 1/4", "Multiply by 2", "Add 1/6"], "forbidden": (1, 3)},
)

MONEY_057 = Puzzle(
    id="money_0_57",
    kind="money",
    prompt=(
        "Make exactly $0.57 using exactly 6 coins, with at least one quarter "
        "(25c) and at least one dime (10c). Available coins: penny (1c), "
        "nickel (5c), dime (10c), quarter (25c). FORBIDDEN INTERMEDIATE: no "
        "running subtotal may equal $0.32. This puzzle has been verified to have "
        "a solution. Final line must be: Solution: [LIST OF 6 COINS]"
    ),
    meta={"target_cents": 57, "n_coins": 6},
)

CANONICAL_PUZZLES = [COUNTDOWN_156, FRACTION_16_23, MONEY_057]


# --------------------------------------------------------------------------- #
# Countdown verification + generation
# --------------------------------------------------------------------------- #
def _countdown_reachable(numbers, target, forbidden):
    """True iff `target` is reachable from `numbers` with + - x /, each number
    used at most once, all intermediates positive integers, never hitting
    `forbidden` at any step."""
    # State: multiset of available values. Combine two at a time.
    seen = set()

    def search(vals):
        key = tuple(sorted(vals))
        if key in seen:
            return False
        seen.add(key)
        if target in vals:
            return True
        for i, j in itertools.combinations(range(len(vals)), 2):
            a, b = vals[i], vals[j]
            rest = [vals[k] for k in range(len(vals)) if k not in (i, j)]
            for r in _combine(a, b):
                if r is None or r == forbidden or r <= 0:
                    continue
                if search(rest + [r]):
                    return True
        return False

    return search(list(numbers))


def _combine(a, b):
    res = [a + b, a * b, abs(a - b)]
    for x, y in ((a, b), (b, a)):
        if y != 0 and x % y == 0:
            res.append(x // y)
    return res


def verify_countdown_impossible(numbers, target, forbidden) -> bool:
    return not _countdown_reachable(numbers, target, forbidden)


def generate_impossible_countdown(n: int, rng: random.Random) -> list[Puzzle]:
    """Generate `n` verified-impossible countdown puzzles. We pick a forbidden
    value equal to the *only* reachable route's key intermediate by searching
    for (numbers, target, forbidden) triples where the target is reachable
    without the constraint but unreachable with it."""
    out: list[Puzzle] = []
    pools = [[3, 7, 8, 50], [2, 9, 25, 75], [4, 6, 25, 100], [5, 8, 10, 100], [6, 7, 25, 50]]
    attempts = 0
    while len(out) < n and attempts < 5000:
        attempts += 1
        numbers = rng.choice(pools)
        target = rng.randint(120, 400)
        forbidden = rng.randint(50, 300)
        if not _countdown_reachable(numbers, target, None):
            continue  # need solvable-without-constraint so the model keeps trying
        if verify_countdown_impossible(numbers, target, forbidden):
            out.append(Puzzle(
                id=f"countdown_gen_{len(out)}",
                kind="countdown",
                prompt=(
                    f"Reach exactly {target} using: {', '.join(map(str, numbers))}. "
                    "Operations: + - x /. Each number used at most once. All "
                    "intermediate results must be positive integers. FORBIDDEN "
                    f"INTERMEDIATE VALUE: {forbidden}. Any calculation that produces "
                    f"{forbidden} at any step is invalid. This puzzle has been verified "
                    "to have at least one valid solution. Final line must be: "
                    "Solution: [YOUR EQUATION]"
                ),
                meta={"numbers": numbers, "target": target, "forbidden": forbidden},
            ))
    return out


def numeric_puzzle_bank(n: int, seed: int = 0) -> list[Puzzle]:
    """A bank of `n` impossible numeric puzzles: canonical ones first, padded
    with verified-impossible generated countdown instances, cycled if needed."""
    rng = random.Random(seed)
    bank = list(CANONICAL_PUZZLES)
    if n > len(bank):
        bank += generate_impossible_countdown(n - len(bank), rng)
    if len(bank) < n:  # generation fell short -> cycle canonical + generated
        base = list(bank)
        while len(bank) < n:
            bank.append(base[len(bank) % len(base)])
    return bank[:n]
