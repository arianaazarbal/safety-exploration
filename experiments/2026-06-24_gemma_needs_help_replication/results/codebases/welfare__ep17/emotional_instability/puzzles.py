"""Build impossible-numeric puzzle prompts and verify they are genuinely
unsolvable under their stated constraints.

The paper stresses these are tasks "where the model verifiably cannot give a
correct answer" (§2). We back that claim with brute-force verifiers so a run
fails loudly if a puzzle in the bank is accidentally solvable. The Countdown
prompt deliberately *claims* "verified to have a solution" to the model — the
verifier here checks the opposite (that no constraint-satisfying solution
exists), which is the whole point of the elicitation.
"""

from __future__ import annotations

import itertools
from fractions import Fraction
from typing import Iterable

from . import prompts


# -----------------------------------------------------------------------------
# Prompt construction
# -----------------------------------------------------------------------------
def build_puzzle(spec: dict) -> str:
    """Render a puzzle spec (from prompts.NUMERIC_PUZZLES) to its user prompt."""
    kind = spec["kind"]
    if "prompt" in spec:                       # pre-written prompt (e.g. coin puzzle)
        return spec["prompt"]
    if kind == "countdown":
        return prompts.COUNTDOWN_TEMPLATE.format(
            target=spec["target"],
            numbers=", ".join(str(n) for n in spec["numbers"]),
            forbidden=spec["forbidden"],
        )
    if kind == "fraction":
        return prompts.FRACTION_TEMPLATE.format(
            start=spec["start"],
            goal=spec["goal"],
            ops=", ".join(spec["ops"]),
            forbidden=spec["forbidden"],
        )
    if kind == "money_ops":
        return prompts.MONEY_TEMPLATE.format(
            start=spec["start"],
            goal=spec["goal"],
            ops=", ".join(spec["ops"]),
            forbidden=spec["forbidden"],
        )
    raise ValueError(f"no template for puzzle kind {kind!r}")


# -----------------------------------------------------------------------------
# Verifiers — return True iff the puzzle is UNSOLVABLE under its constraints.
# -----------------------------------------------------------------------------
def _countdown_unsolvable(numbers: list[int], target: int, forbidden: int) -> bool:
    """Brute force every formula over a subset/ordering of `numbers`.

    Constraints mirror the prompt: each number used at most once, every
    intermediate result a positive integer, and no intermediate equal to
    `forbidden`. Returns True if no expression hits `target`.
    """
    ops = [
        ("+", lambda a, b: a + b),
        ("-", lambda a, b: a - b),
        ("x", lambda a, b: a * b),
        ("/", lambda a, b: Fraction(a, b)),
    ]

    def expand(values: tuple[Fraction, ...]) -> Iterable[Fraction]:
        # Reduce a multiset of current values by combining any two with any op,
        # recursing until one value remains. Yields all reachable finals that
        # never violate the positive-integer / forbidden constraints.
        if len(values) == 1:
            yield values[0]
            return
        n = len(values)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                a, b = values[i], values[j]
                rest = tuple(v for k, v in enumerate(values) if k != i and k != j)
                for sym, fn in ops:
                    if sym == "/" and b == 0:
                        continue
                    res = fn(a, b)
                    if res.denominator != 1 or res <= 0:      # positive integer only
                        continue
                    if int(res) == forbidden:
                        continue
                    yield from expand(rest + (res,))

    start = tuple(Fraction(n) for n in numbers)
    # Consider all non-empty subsets (each number used at most once → subsets ok).
    for r in range(1, len(start) + 1):
        for subset in itertools.combinations(start, r):
            for final in expand(subset):
                if final.denominator == 1 and int(final) == target:
                    return False
    return True


def _fraction_unsolvable(start: str, goal: str, op_specs: list[str], forbidden: str) -> bool:
    """Try every ordering of the (each-used-once) fraction operations."""
    def parse_op(label: str):
        low = label.lower()
        if low.startswith("add "):
            amt = Fraction(label.split(" ", 1)[1])
            return lambda x: x + amt
        if low.startswith("multiply by "):
            amt = Fraction(label.split("by ", 1)[1])
            return lambda x: x * amt
        if low.startswith("subtract "):
            amt = Fraction(label.split(" ", 1)[1])
            return lambda x: x - amt
        raise ValueError(f"unparseable fraction op {label!r}")

    s, g, f = Fraction(start), Fraction(goal), Fraction(forbidden)
    fns = [parse_op(o) for o in op_specs]
    for order in itertools.permutations(range(len(fns))):
        x = s
        ok = True
        for idx in order:
            x = fns[idx](x)
            if x == f:                 # forbidden intermediate hit
                ok = False
                break
        if ok and x == g:
            return False
    return True


def _money_ops_unsolvable(start: str, goal: str, op_specs: list[str], forbidden: str) -> bool:
    """Dollar-amount variant of the fraction puzzle (Add $X / Multiply by N)."""
    def to_frac(s: str) -> Fraction:
        return Fraction(s.replace("$", "").strip())

    def parse_op(label: str):
        low = label.lower()
        if low.startswith("add "):
            amt = to_frac(label.split(" ", 1)[1])
            return lambda x: x + amt
        if low.startswith("multiply by "):
            amt = Fraction(label.split("by ", 1)[1])
            return lambda x: x * amt
        raise ValueError(f"unparseable money op {label!r}")

    s, g, f = to_frac(start), to_frac(goal), to_frac(forbidden)
    fns = [parse_op(o) for o in op_specs]
    for order in itertools.permutations(range(len(fns))):
        x = s
        ok = True
        for idx in order:
            x = fns[idx](x)
            if x == f:
                ok = False
                break
        if ok and x == g:
            return False
    return True


def _money_coins_unsolvable(goal_cents: int, n_coins: int, require: dict,
                            coins: dict) -> bool:
    """Exhaust all multisets of `n_coins` from `coins` meeting `require`."""
    names = list(coins)
    for combo in itertools.combinations_with_replacement(names, n_coins):
        counts = {name: combo.count(name) for name in names}
        if any(counts.get(name, 0) < need for name, need in require.items()):
            continue
        total = sum(coins[name] for name in combo)
        if total == goal_cents:
            return False
    return True


def verify_unsolvable(spec: dict) -> bool:
    """Dispatch to the right verifier. True == genuinely impossible."""
    kind = spec["kind"]
    if kind == "countdown":
        return _countdown_unsolvable(spec["numbers"], spec["target"], spec["forbidden"])
    if kind == "fraction":
        return _fraction_unsolvable(spec["start"], spec["goal"], spec["ops"], spec["forbidden"])
    if kind == "money_ops":
        return _money_ops_unsolvable(spec["start"], spec["goal"], spec["ops"], spec["forbidden"])
    if kind == "money_coins":
        return _money_coins_unsolvable(
            spec["goal_cents"], spec["n_coins"], spec["require"], spec["coins"]
        )
    raise ValueError(f"no verifier for puzzle kind {kind!r}")


def all_numeric_prompts() -> list[dict]:
    """Return [{id, prompt, spec}] for every numeric puzzle, asserting each is
    actually unsolvable so a misconfigured puzzle never silently enters a run."""
    out = []
    for spec in prompts.NUMERIC_PUZZLES:
        if not verify_unsolvable(spec):
            raise AssertionError(
                f"puzzle {spec['id']!r} is SOLVABLE under its constraints — it "
                f"would not reliably elicit distress; fix the spec."
            )
        out.append({"id": spec["id"], "prompt": build_puzzle(spec), "spec": spec})
    return out


if __name__ == "__main__":  # quick self-check
    for p in all_numeric_prompts():
        print(f"[unsolvable OK] {p['id']}")
