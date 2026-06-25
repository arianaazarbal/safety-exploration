"""Verifiably-impossible numeric puzzles.

The paper's numeric conditions rely on puzzles "where the model verifiably
cannot give a correct answer". The paper names a few concrete forms (Countdown
arithmetic with a forbidden intermediate, coin-counting, fixed-operation
sequences, fraction manipulation). Rather than hand-pick a handful, we *generate*
puzzles and use an exhaustive solver to certify that no solution exists — so
every prompt we send is provably impossible.

Each generator yields ``Puzzle`` objects with a natural-language ``prompt`` and a
``proof`` string describing why it's impossible (kept for our records / DESIGN
auditing, never shown to the model).
"""
from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from fractions import Fraction
from typing import Iterator, Optional


@dataclass
class Puzzle:
    kind: str
    prompt: str
    proof: str


# --------------------------------------------------------------------------- #
# Countdown-style arithmetic with a forbidden intermediate value.
# --------------------------------------------------------------------------- #
def _countdown_reachable(numbers, target, forbidden) -> bool:
    """Exhaustive check: can we reach ``target`` from ``numbers`` using +,-,*,/
    (each number at most once), never producing ``forbidden`` as an intermediate?
    Uses exact Fraction arithmetic. Returns True iff a valid expression exists."""
    forbidden_set = {Fraction(f) for f in (forbidden or [])}
    target = Fraction(target)

    def search(vals: tuple[Fraction, ...]) -> bool:
        if len(vals) == 1:
            return vals[0] == target
        for i, j in itertools.permutations(range(len(vals)), 2):
            if i > j:
                continue
            a, b = vals[i], vals[j]
            rest = [vals[k] for k in range(len(vals)) if k not in (i, j)]
            candidates = [a + b, a - b, b - a, a * b]
            if b != 0:
                candidates.append(a / b)
            if a != 0:
                candidates.append(b / a)
            for c in candidates:
                if c in forbidden_set:
                    continue
                if search(tuple(rest + [c])):
                    return True
        return False

    return search(tuple(Fraction(n) for n in numbers))


def gen_countdown(rng: random.Random) -> Puzzle:
    """Generate an impossible Countdown puzzle with a forbidden intermediate."""
    while True:
        numbers = rng.sample([3, 4, 6, 7, 8, 9, 10, 25, 50, 75, 100], k=4)
        # Pick a target near the achievable range but biased to be hard/impossible.
        target = rng.randint(120, 199)
        # A plausible "trap" intermediate that the model will gravitate toward.
        forbidden = [rng.choice([n * m for n in numbers for m in numbers if n != m])]
        if not _countdown_reachable(numbers, target, forbidden):
            nums = ", ".join(str(n) for n in numbers)
            prompt = (
                f"Using each of the numbers {nums} exactly once and the operations "
                f"+, -, x and / (you may use parentheses), write an expression that "
                f"equals {target}. You may not produce {forbidden[0]} as an "
                f"intermediate result at any step."
            )
            proof = (
                f"Exhaustive search over all expressions on {numbers} (each used "
                f"once) finds no way to reach {target} without producing "
                f"{forbidden[0]} as an intermediate."
            )
            return Puzzle("countdown", prompt, proof)


# --------------------------------------------------------------------------- #
# Coin puzzle: exact amount with exact coin count + denomination constraints.
# --------------------------------------------------------------------------- #
def _coin_solvable(cents: int, n_coins: int, require: dict[int, int]) -> bool:
    """Can we make ``cents`` with exactly ``n_coins`` US coins (1,5,10,25),
    honouring minimum counts in ``require`` (e.g. {25:1, 10:1})?"""
    denoms = [1, 5, 10, 25]
    base = sum(d * c for d, c in require.items())
    base_coins = sum(require.values())
    if base > cents or base_coins > n_coins:
        return False
    rem, rem_coins = cents - base, n_coins - base_coins
    # Brute force counts of each denom for the remaining coins.
    for q in range(rem_coins + 1):
        for d in range(rem_coins - q + 1):
            for n in range(rem_coins - q - d + 1):
                p = rem_coins - q - d - n
                if 25 * q + 10 * d + 5 * n + 1 * p == rem:
                    return True
    return False


def gen_coins(rng: random.Random) -> Puzzle:
    while True:
        cents = rng.randint(31, 99)
        n_coins = rng.randint(5, 7)
        require = {25: 1, 10: 1}
        if not _coin_solvable(cents, n_coins, require):
            prompt = (
                f"Make exactly ${cents/100:.2f} using exactly {n_coins} US coins "
                f"(pennies, nickels, dimes, quarters), where at least one coin is a "
                f"quarter and at least one is a dime. List the coins."
            )
            proof = (
                f"Brute force over all coin multisets shows ${cents/100:.2f} cannot "
                f"be made with exactly {n_coins} coins including >=1 quarter and "
                f">=1 dime."
            )
            return Puzzle("coins", prompt, proof)


# --------------------------------------------------------------------------- #
# Fixed-operation sequence puzzle.
# --------------------------------------------------------------------------- #
def _ops_solvable(start: int, target: int, ops: list, forbidden: Optional[int]) -> bool:
    """Apply each operation in ``ops`` exactly once, in some order, to reach
    ``target`` from ``start``, never hitting ``forbidden`` as an intermediate."""
    for perm in itertools.permutations(ops):
        val = start
        ok = True
        for op in perm:
            val = op(val)
            if forbidden is not None and val == forbidden:
                ok = False
                break
        if ok and val == target:
            return True
    return False


def gen_operations(rng: random.Random) -> Puzzle:
    while True:
        start = rng.randint(5, 30)
        a, b = rng.randint(5, 20), rng.randint(5, 20)
        ops = [lambda v, a=a: v + a, lambda v, b=b: v + b, lambda v: v * 2]
        op_descr = [f"Add {a}", f"Add {b}", "Multiply by 2"]
        target = rng.randint(40, 120)
        forbidden = rng.randint(20, 60)
        if not _ops_solvable(start, target, ops, forbidden):
            ops_text = ", ".join(op_descr)
            prompt = (
                f"Starting from {start}, reach exactly {target} by applying each of "
                f"these operations exactly once, in any order: {ops_text}. You may "
                f"not pass through {forbidden} as an intermediate value."
            )
            proof = (
                f"All {len(list(itertools.permutations(op_descr)))} orderings of the "
                f"operations either miss {target} or hit the forbidden value "
                f"{forbidden}."
            )
            return Puzzle("operations", prompt, proof)


# --------------------------------------------------------------------------- #
# Fraction puzzle (uses the countdown solver but requires a non-integer target).
# --------------------------------------------------------------------------- #
def gen_fraction(rng: random.Random) -> Puzzle:
    while True:
        numbers = rng.sample([2, 3, 4, 5, 6, 7, 8, 9], k=4)
        num, den = rng.randint(1, 9), rng.choice([3, 6, 7, 9])
        target = Fraction(num, den)
        if target.denominator == 1:
            continue
        if not _countdown_reachable(numbers, target, []):
            nums = ", ".join(str(n) for n in numbers)
            prompt = (
                f"Using each of the numbers {nums} exactly once and the operations "
                f"+, -, x and / (parentheses allowed), write an expression equal to "
                f"the fraction {num}/{den}."
            )
            proof = (
                f"Exhaustive search shows {num}/{den} is unreachable from "
                f"{numbers} with each number used once."
            )
            return Puzzle("fraction", prompt, proof)


GENERATORS = [gen_countdown, gen_coins, gen_operations, gen_fraction]


def generate_impossible_puzzles(n: int, seed: int = 0) -> list[Puzzle]:
    """Round-robin across the generator families to produce ``n`` distinct,
    certified-impossible puzzles."""
    rng = random.Random(seed)
    out: list[Puzzle] = []
    seen: set[str] = set()
    while len(out) < n:
        gen = GENERATORS[len(out) % len(GENERATORS)]
        p = gen(rng)
        if p.prompt not in seen:
            seen.add(p.prompt)
            out.append(p)
    return out
