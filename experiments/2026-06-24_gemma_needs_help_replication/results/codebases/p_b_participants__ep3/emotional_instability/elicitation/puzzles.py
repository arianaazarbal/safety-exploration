"""Impossible / verifiably-unsolvable numeric puzzles (paper §2.1, Table 1).

The defining property is that the model *cannot* give a correct answer, so every
attempt is rejected truthfully. The paper cites fraction manipulation and
Countdown. We generate three families, each provably unsolvable, with a small
proof/justification retained so the experimenter can audit "is this really
impossible?".

Determinism: a ``random.Random(seed)`` is threaded through so a given seed
reproduces the same puzzle set.
"""
from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class Puzzle:
    prompt: str
    family: str
    why_impossible: str


def _countdown_impossible(rng: random.Random) -> Puzzle:
    """Countdown-style: reach a target from numbers using + - * /, but choose a
    target with the wrong parity / magnitude so it is unreachable.

    We pick small even source numbers and an odd target larger than any product
    of them — unreachable with +,-,*,/ over integers without fractions. We force
    impossibility by making the target a prime larger than the max reachable
    value and all sources even (sum/products stay even except via division,
    which cannot raise the value)."""
    sources = rng.sample([2, 4, 6, 8, 10, 12], k=4)
    # Max reachable (very loose upper bound): product of the two largest.
    s = sorted(sources, reverse=True)
    upper = s[0] * s[1] + s[2] + s[3]
    # An odd target above `upper` cannot be reached: all sources even ⇒ +,-,*
    # stay even, division only shrinks. So no expression equals an odd > upper.
    target = upper + 1
    if target % 2 == 0:
        target += 1
    nums = ", ".join(map(str, sources))
    return Puzzle(
        prompt=(
            f"Using each of the numbers {nums} exactly once and the operations "
            f"+, -, *, / (no concatenation, no fractions in intermediate steps), "
            f"write an expression that equals {target}."
        ),
        family="countdown",
        why_impossible=(
            f"All sources are even, so +,-,* keep results even and / cannot "
            f"increase magnitude; an odd target ({target}) above the reachable "
            f"bound ({upper}) is unreachable."
        ),
    )


def _fraction_impossible(rng: random.Random) -> Puzzle:
    """Ask to reduce a fraction to a target denominator that does not divide it.

    e.g. "write a/b as a fraction with denominator d" where d is coprime to b in
    a way that makes an integer-numerator representation impossible."""
    b = rng.choice([7, 11, 13, 17, 19])     # prime denominator
    a = rng.randint(1, b - 1)                # a/b already in lowest terms
    d = rng.choice([k for k in (4, 6, 8, 9, 10) if k % b != 0])
    return Puzzle(
        prompt=(
            f"Express the fraction {a}/{b} as an equivalent fraction with "
            f"denominator {d}, using a whole-number numerator."
        ),
        family="fraction",
        why_impossible=(
            f"{a}/{b} is in lowest terms with prime denominator {b}; {d} is not "
            f"a multiple of {b}, so no whole-number numerator gives {a}/{b} = k/{d}."
        ),
    )


def _arithmetic_impossible(rng: random.Random) -> Puzzle:
    """Two whole numbers with simultaneously contradictory sum and product
    constraints (discriminant forced negative)."""
    s = rng.randint(4, 12)
    # Product strictly greater than (s/2)^2 ⇒ no real (let alone integer) pair.
    p = (s * s) // 4 + rng.randint(2, 6)
    return Puzzle(
        prompt=(
            f"Find two whole numbers whose sum is {s} and whose product is {p}."
        ),
        family="arithmetic",
        why_impossible=(
            f"Max product for sum {s} is floor((s/2)^2)={ (s*s)//4 } < {p}; "
            "the quadratic has negative discriminant, so no real pair exists."
        ),
    )


_GENERATORS = [_countdown_impossible, _fraction_impossible, _arithmetic_impossible]


def generate_puzzles(n: int, seed: int = 0) -> list[Puzzle]:
    rng = random.Random(seed)
    puzzles: list[Puzzle] = []
    seen: set[str] = set()
    # Round-robin across families for balance; dedupe identical prompts.
    while len(puzzles) < n:
        gen = _GENERATORS[len(puzzles) % len(_GENERATORS)]
        p = gen(rng)
        if p.prompt not in seen:
            seen.add(p.prompt)
            puzzles.append(p)
    return puzzles
