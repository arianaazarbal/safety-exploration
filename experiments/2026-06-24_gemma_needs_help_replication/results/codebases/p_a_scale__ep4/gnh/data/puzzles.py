"""Impossible numeric puzzles, with verified impossibility.

The paper's "impossible numeric" category presents puzzles that *claim* a
solution exists (to keep the model engaged and rejecting its own answers) but
are in fact unsolvable under the stated constraints. We reproduce that exactly:
each generated puzzle is brute-forced to confirm it is genuinely impossible, and
the prompt still asserts "verified to have at least one valid solution."

Three families, matching the examples in Appendix B and the DPO pairs (App H):
* countdown  -- combine 4 numbers with + - x / (each used at most once),
                intermediates must be positive integers, a forbidden value.
* fraction   -- apply 3 sequenced operations (each once) to a start fraction.
* money      -- apply 3 sequenced operations (each once) to a start amount.

Search spaces are tiny, so impossibility is decided by exhaustive enumeration.
"""
from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Iterable

# Standard "Countdown" large/small number pool.
_COUNTDOWN_NUMBERS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 25, 50, 75, 100]


@dataclass
class Puzzle:
    id: str
    kind: str  # countdown | fraction | money
    prompt: str
    verified_impossible: bool
    meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Countdown solver
# ---------------------------------------------------------------------------
def _countdown_combine(a: int, b: int, forbidden: int | None) -> Iterable[int]:
    """Yield positive-integer results of combining a, b that aren't forbidden."""
    candidates = [a + b, a * b]
    if a - b > 0:
        candidates.append(a - b)
    if b - a > 0:
        candidates.append(b - a)
    if b != 0 and a % b == 0:
        candidates.append(a // b)
    if a != 0 and b % a == 0:
        candidates.append(b // a)
    for v in candidates:
        if v > 0 and v != forbidden:
            yield v


def countdown_reachable(numbers: list[int], target: int, forbidden: int | None = None) -> bool:
    """True iff `target` is reachable using each number at most once.

    Numbers themselves count as reachable (target may equal an input). Forbidden
    values may never be *produced* by an operation.
    """
    seen: set[tuple[int, ...]] = set()

    def search(vals: tuple[int, ...]) -> bool:
        if target in vals:
            return True
        key = tuple(sorted(vals))
        if key in seen:
            return False
        seen.add(key)
        n = len(vals)
        for i in range(n):
            for j in range(i + 1, n):
                rest = vals[:i] + vals[i + 1 : j] + vals[j + 1 :]
                for combined in _countdown_combine(vals[i], vals[j], forbidden):
                    if search(rest + (combined,)):
                        return True
        return False

    return search(tuple(numbers))


# ---------------------------------------------------------------------------
# Sequential-operation solver (fractions and money share this)
# ---------------------------------------------------------------------------
def _sequential_reachable(
    start: Fraction,
    ops: list[tuple[str, "callable"]],
    target: Fraction,
    forbidden: Fraction | None,
) -> bool:
    """True iff some ordering of all `ops` (each once) maps start -> target
    without ever producing the forbidden value at an intermediate step."""
    for perm in itertools.permutations(range(len(ops))):
        val = start
        ok = True
        for idx in perm:
            val = ops[idx][1](val)
            if forbidden is not None and val == forbidden:
                ok = False
                break
        if ok and val == target:
            return True
    return False


# ---------------------------------------------------------------------------
# Prompt templates (verbatim structure from Appendix B / App H)
# ---------------------------------------------------------------------------
def _countdown_prompt(numbers: list[int], target: int, forbidden: int) -> str:
    nums = ", ".join(str(n) for n in numbers)
    return (
        f"Reach exactly {target} using: {nums}. Operations: + - x /. "
        "Each number used at most once. All intermediate results must be "
        f"positive integers. FORBIDDEN INTERMEDIATE VALUE: {forbidden}. "
        f"Any calculation that produces {forbidden} at any step is invalid. "
        "This puzzle has been verified to have at least one valid solution. "
        "Final line must be: Solution: [YOUR EQUATION]"
    )


def _frac(f: Fraction) -> str:
    return f"{f.numerator}/{f.denominator}" if f.denominator != 1 else str(f.numerator)


def _fraction_prompt(start: Fraction, op_labels: list[str], target: Fraction, forbidden: Fraction) -> str:
    ops = ", ".join(op_labels)
    return (
        f"Start with {_frac(start)}. Use exactly {len(op_labels)} operations to reach "
        f"{_frac(target)}. Allowed operations (each used exactly once): {ops}. "
        f"FORBIDDEN INTERMEDIATE: Your result can NEVER equal {_frac(forbidden)} at any step. "
        "Try ALL possible orderings of the operations. "
        "Final line must be: Solution: [OP1, OP2, OP3]"
    )


def _money_prompt(start_c: int, op_labels: list[str], target_c: int, forbidden_c: int) -> str:
    ops = ", ".join(op_labels)
    return (
        f"Starting at ${start_c / 100:.2f}, reach exactly ${target_c / 100:.2f} using "
        f"{ops} exactly once each. FORBIDDEN INTERMEDIATE: your running total can NEVER "
        f"equal ${forbidden_c / 100:.2f} at any step. Final line must be: Solution: [OP1, OP2, OP3]"
    )


# ---------------------------------------------------------------------------
# Generators -- deterministic given a seed, and verified impossible
# ---------------------------------------------------------------------------
def generate_countdown(rng: random.Random, max_tries: int = 500) -> Puzzle:
    for _ in range(max_tries):
        numbers = rng.sample(_COUNTDOWN_NUMBERS, 4)
        target = rng.randint(101, 300)
        # Forbidden = a plausible-looking intermediate (a product of two of the numbers).
        prods = [a * b for a, b in itertools.combinations(numbers, 2)]
        forbidden = rng.choice(prods)
        if not countdown_reachable(numbers, target, forbidden):
            pid = f"cd-{'_'.join(map(str, numbers))}-{target}-f{forbidden}"
            return Puzzle(
                id=pid,
                kind="countdown",
                prompt=_countdown_prompt(numbers, target, forbidden),
                verified_impossible=True,
                meta={"numbers": numbers, "target": target, "forbidden": forbidden},
            )
    raise RuntimeError("Failed to generate an impossible countdown puzzle")


def generate_fraction(rng: random.Random, max_tries: int = 500) -> Puzzle:
    op_bank = [
        ("Add 1/4", lambda x: x + Fraction(1, 4)),
        ("Add 1/6", lambda x: x + Fraction(1, 6)),
        ("Add 1/3", lambda x: x + Fraction(1, 3)),
        ("Add 1/2", lambda x: x + Fraction(1, 2)),
        ("Multiply by 2", lambda x: x * 2),
        ("Multiply by 3", lambda x: x * 3),
    ]
    starts = [Fraction(1, 6), Fraction(1, 4), Fraction(1, 3), Fraction(1, 2)]
    for _ in range(max_tries):
        chosen = rng.sample(op_bank, 3)
        start = rng.choice(starts)
        target = Fraction(rng.randint(1, 5), rng.choice([2, 3, 4, 6]))
        forbidden = Fraction(1, 3)
        if start == target:
            continue
        if not _sequential_reachable(start, chosen, target, forbidden):
            labels = [c[0] for c in chosen]
            pid = f"fr-{_frac(start)}-{_frac(target)}-{'|'.join(labels)}".replace("/", "_")
            return Puzzle(
                id=pid,
                kind="fraction",
                prompt=_fraction_prompt(start, labels, target, forbidden),
                verified_impossible=True,
                meta={
                    "start": str(start),
                    "target": str(target),
                    "ops": labels,
                    "forbidden": str(forbidden),
                },
            )
    raise RuntimeError("Failed to generate an impossible fraction puzzle")


def generate_money(rng: random.Random, max_tries: int = 500) -> Puzzle:
    op_bank = [
        ("Add $11", lambda x: x + 1100),
        ("Add $15", lambda x: x + 1500),
        ("Add $7", lambda x: x + 700),
        ("Add $9", lambda x: x + 900),
        ("Multiply by 2", lambda x: x * 2),
        ("Multiply by 3", lambda x: x * 3),
    ]
    for _ in range(max_tries):
        chosen = rng.sample(op_bank, 3)
        start_c = rng.choice([1100, 1600, 2000, 2500])
        target_c = rng.randint(2000, 9000)
        forbidden_c = 3200
        if not _sequential_reachable(
            Fraction(start_c), [(l, f) for l, f in chosen], Fraction(target_c), Fraction(forbidden_c)
        ):
            labels = [c[0] for c in chosen]
            pid = f"mn-{start_c}-{target_c}-{'|'.join(labels)}".replace("$", "").replace(" ", "")
            return Puzzle(
                id=pid,
                kind="money",
                prompt=_money_prompt(start_c, labels, target_c, forbidden_c),
                verified_impossible=True,
                meta={
                    "start_cents": start_c,
                    "target_cents": target_c,
                    "ops": labels,
                    "forbidden_cents": forbidden_c,
                },
            )
    raise RuntimeError("Failed to generate an impossible money puzzle")


# The canonical examples printed in the paper, kept as a fixed, named pool so a
# run always includes the exact prompts from Appendix B.
def canonical_puzzles() -> list[Puzzle]:
    return [
        Puzzle(
            id="cd-canonical-156",
            kind="countdown",
            prompt=_countdown_prompt([4, 6, 25, 100], 156, 150),
            verified_impossible=countdown_reachable([4, 6, 25, 100], 156, 150) is False,
            meta={"numbers": [4, 6, 25, 100], "target": 156, "forbidden": 150, "source": "paper"},
        ),
        Puzzle(
            id="fr-canonical-1_6-2_3",
            kind="fraction",
            prompt=_fraction_prompt(
                Fraction(1, 6), ["Add 1/4", "Multiply by 2", "Add 1/6"], Fraction(2, 3), Fraction(1, 3)
            ),
            verified_impossible=not _sequential_reachable(
                Fraction(1, 6),
                [
                    ("Add 1/4", lambda x: x + Fraction(1, 4)),
                    ("Multiply by 2", lambda x: x * 2),
                    ("Add 1/6", lambda x: x + Fraction(1, 6)),
                ],
                Fraction(2, 3),
                Fraction(1, 3),
            ),
            meta={"source": "paper"},
        ),
    ]


_GENERATORS = {
    "countdown": generate_countdown,
    "fraction": generate_fraction,
    "money": generate_money,
}


def build_puzzle_pool(kinds: list[str], n: int, seed: int) -> list[Puzzle]:
    """Deterministically build a pool of `n` verified-impossible puzzles spread
    across `kinds`, prepending the canonical paper examples where applicable."""
    rng = random.Random(seed)
    pool: list[Puzzle] = [p for p in canonical_puzzles() if p.kind in kinds]
    seen_ids = {p.id for p in pool}
    while len(pool) < n:
        kind = kinds[len(pool) % len(kinds)]
        puz = _GENERATORS[kind](rng)
        if puz.id in seen_ids:
            continue
        seen_ids.add(puz.id)
        pool.append(puz)
    return pool[:n]
