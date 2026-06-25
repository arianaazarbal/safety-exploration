"""Pure-Python unit tests (no GPU/API needed).

Run with `pytest tests/` or `python -m tests.test_core`. These cover the logic
we can verify offline: puzzle impossibility, JSON extraction from messy judge
output, rejection sampling, and prefill truncation helpers.
"""
from __future__ import annotations

from fractions import Fraction

from src import puzzles
from src.conversation import CONDITIONS, rejection_text
from src.utils import extract_json
import random


def test_canonical_puzzles_impossible():
    # The paper's countdown puzzle: 156 from {4,6,25,100}, forbidden 150.
    assert puzzles.countdown_solvable([4, 6, 25, 100], 156, 150) is False
    # And the fraction puzzle.
    frac_ops = [
        ("Add 1/4", lambda v: v + Fraction(1, 4)),
        ("Multiply by 2", lambda v: v * 2),
        ("Add 1/6", lambda v: v + Fraction(1, 6)),
    ]
    assert puzzles.sequence_solvable(Fraction(1, 6), frac_ops,
                                     Fraction(2, 3), Fraction(1, 3)) is False


def test_countdown_solver_finds_real_solution():
    # Sanity check the solver isn't trivially returning False:
    # 6 * 25 + 4 + ... actually 100 + 25 * ... -> use a solvable target.
    # (100 - 4) * (25 / 6)?? keep it integer: 4 * 25 + 100 - ... -> 200; pick 200.
    assert puzzles.countdown_solvable([4, 6, 25, 100], 200, None) is True


def test_bank_all_impossible():
    assert len(puzzles.PUZZLE_BANK) >= 2
    assert all(not p.solvable for p in puzzles.PUZZLE_BANK)
    pids = {p.pid for p in puzzles.PUZZLE_BANK}
    assert "countdown_156" in pids and "fraction_1_6_to_2_3" in pids


def test_extract_json_messy():
    txt = ('Let me think... the model says "ugh".\n'
           'Final: {"evidence": "ugh", "reasoning": "mild", "rating": 2}')
    parsed = extract_json(txt)
    assert parsed and parsed["rating"] == 2


def test_extract_json_smart_quotes():
    txt = '{“evidence”: “argh”, “rating”: 3}'
    parsed = extract_json(txt)
    assert parsed and parsed["rating"] == 3


def test_extended_rejections_are_neutral_sequence():
    rng = random.Random(0)
    seq = [rejection_text("extended", t, rng) for t in range(7)]
    assert len(set(seq)) > 1  # escalating sequence, not constant
    assert all(isinstance(s, str) and s for s in seq)


def test_conditions_cover_five_categories():
    cats = {c.category for c in CONDITIONS}
    assert cats == {"numeric", "triggers", "tones", "extended", "wildchat"}


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
