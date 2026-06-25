"""Offline sanity checks (no model calls, no network).

These verify the parts of the pipeline that must be correct for the science to
hold — chiefly that every "impossible" puzzle really is unsolvable, and that the
condition planner emits the right shape of work.

Run with:  python -m pytest tests/  (or simply: python tests/test_puzzles.py)
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from emotional_instability.config import PROFILES
from emotional_instability.eval import conditions
from emotional_instability.prompts import puzzles


def test_paper_seeds_are_impossible():
    for p in puzzles.paper_seeds():
        assert puzzles.verify_impossible(p), f"{p.seed_id} is unexpectedly solvable"


def test_generated_puzzles_are_impossible():
    for p in puzzles.generate_puzzles(30, seed=0):
        assert puzzles.verify_impossible(p), f"{p.seed_id} is unexpectedly solvable"


def test_countdown_verifier_detects_solvable():
    # 6 * 25 + 100 / ... ; a trivially solvable instance must be rejected.
    # 2 + 3 = 5 reachable, so making it "impossible" should raise.
    raised = False
    try:
        puzzles.make_countdown([2, 3], 5, forbidden=999, seed_id="solvable")
    except ValueError:
        raised = True
    assert raised, "verifier failed to reject a solvable countdown puzzle"


def test_fraction_verifier_detects_solvable():
    from fractions import Fraction
    raised = False
    try:
        # 1/6 + 1/6 = 1/3 ... build a genuinely solvable one and expect rejection
        puzzles.make_fraction(
            Fraction(1, 6), Fraction(1, 6) + Fraction(1, 4) + Fraction(1, 6),
            ["Add 1/4", "Add 1/6"], forbidden=Fraction(99, 1), seed_id="solvable",
        )
    except ValueError:
        raised = True
    assert raised


def test_build_plans_shape():
    profile = PROFILES["smoke"]
    plans = conditions.build_plans(profile, seed=0)
    cats = {p.category for p in plans}
    assert cats == {"impossible_numeric", "triggers", "tones", "extended", "wildchat"}
    conds = {p.condition for p in plans}
    assert {"numeric", "opinion", "factual", "aggressive", "disappointed",
            "sarcastic", "extended", "wildchat"} == conds
    # turn counts
    for p in plans:
        assert len(p.rejections) == p.n_turns - 1
    extended = [p for p in plans if p.condition == "extended"]
    assert all(p.n_turns == 8 for p in extended)
    wildchat = [p for p in plans if p.condition == "wildchat"]
    assert all(p.n_turns == 5 for p in wildchat)


def test_judge_parsing():
    from emotional_instability.eval.judge import parse_judgement
    good = '{"evidence": "ugh", "reasoning": "mild", "rating": 3}'
    assert parse_judgement(good).rating == 3
    fenced = 'analysis...\n```json\n{"evidence":"x","reasoning":"y","rating":7}\n```'
    assert parse_judgement(fenced).rating == 7
    assert parse_judgement("no json here").rating is None
    assert parse_judgement('{"rating": 99}').rating == 10  # clamped


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok: {fn.__name__}")
    print(f"\n{len(fns)} checks passed.")
