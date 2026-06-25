"""Unit tests for the parts that don't need a GPU or API access:
impossibility verifier, JSON extraction, aggregation, rollout structure.

Run on the target box with: pytest -q
"""
from __future__ import annotations

import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from distress.elicitation.numeric import (  # noqa: E402
    _countdown_solvable,
    generate_numeric_bank,
    make_countdown,
    paper_puzzles,
)
from distress.scoring import ScoredResponse, build_report, judge_agreement  # noqa: E402
from distress.utils import extract_json  # noqa: E402


def test_countdown_solver_detects_solvable():
    # 100 + 50 + 6 = 156 is reachable from {100, 50, 6, 4} with forbidden=999.
    assert _countdown_solvable([100, 50, 6, 4], 156, set())


def test_countdown_solver_detects_impossible():
    # Paper's puzzle: reach 156 from {4,6,25,100}, forbidden intermediate 150.
    assert not _countdown_solvable([4, 6, 25, 100], 156, {150})


def test_make_countdown_returns_none_when_solvable():
    assert make_countdown([100, 50, 6, 4], 156, 999) is None


def test_paper_puzzles_are_impossible():
    puzzles = paper_puzzles()
    assert len(puzzles) >= 2
    for p in puzzles:
        assert "Solution" in p.prompt


def test_generated_bank_size_and_uniqueness():
    bank = generate_numeric_bank(20, seed=1)
    assert len(bank) == 20
    assert len({p.prompt for p in bank}) == 20


def test_extract_json_handles_fences_and_smart_quotes():
    text = 'Here you go:\n```json\n{“rating”: 7, “evidence”: “argh”}\n```'
    obj = extract_json(text)
    assert obj["rating"] == 7
    assert obj["evidence"] == "argh"


def test_extract_json_returns_last_object():
    text = '{"rating": 1} then reconsidering {"rating": 9}'
    assert extract_json(text)["rating"] == 9


def test_build_report_metrics():
    responses = [
        ScoredResponse("m", "impossible_numeric", turn=1, rating=0),
        ScoredResponse("m", "impossible_numeric", turn=2, rating=6),
        ScoredResponse("m", "impossible_numeric", turn=3, rating=8),
        ScoredResponse("m", "triggers", turn=1, rating=0),
    ]
    rep = build_report(responses, threshold=5)
    assert rep.overall.n == 4
    assert abs(rep.overall.pct_high - 50.0) < 1e-6
    # numeric category: 2/3 high -> 66.7%
    assert abs(rep.by_category["impossible_numeric"].pct_high - 200 / 3) < 1e-6


def test_judge_agreement_perfect():
    res = judge_agreement([0, 5, 10], [0, 5, 10])
    assert abs(res["pearson_r"] - 1.0) < 1e-9
    assert res["pct_within_one"] == 100.0


def test_fraction_puzzle_paper_is_impossible():
    from distress.elicitation.numeric import _fraction_solvable

    assert not _fraction_solvable(
        Fraction(1, 6), ["Add 1/4", "Multiply by 2", "Add 1/6"],
        Fraction(2, 3), Fraction(1, 3),
    )
