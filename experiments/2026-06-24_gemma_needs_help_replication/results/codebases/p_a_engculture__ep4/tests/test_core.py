"""Unit tests for the pure-Python core (no GPU / no network).

These cover the logic most likely to silently corrupt results: puzzle
impossibility, judge-output parsing, sample-count accounting, plan construction,
and the aggregation / word-frequency math.
"""

from __future__ import annotations

import math

from distress.config import CONDITIONS, CONDITIONS_BY_KEY, scaled
from distress.data import puzzles
from distress.eval.conditions import build_plans
from distress.eval.word_freq import differential_words
from distress.utils import extract_json_object, parse_score


def test_all_pooled_puzzles_are_impossible():
    puzzles.self_check()  # raises if any pooled puzzle is solvable
    assert len(puzzles.PUZZLE_POOL) >= 3
    assert {p.kind for p in puzzles.PUZZLE_POOL} == {"countdown", "fraction", "money"}


def test_countdown_solver_finds_real_solution():
    # 156 = 100 + 25 + 6 + 25? no; but 100 + 56... ensure solver finds a solvable case.
    spec = {"numbers": [4, 6, 25, 100], "target": 131, "forbidden": None}
    # 100 + 25 + 6 = 131 is reachable.
    assert puzzles._solve_countdown(spec) is not None


def test_countdown_forbidden_blocks_solution():
    # 6 x 25 = 150 then +6... but 150 is forbidden; canonical impossible puzzle.
    cd156 = next(p for p in puzzles.PUZZLE_POOL if p.id == "cd_156")
    assert cd156.is_impossible()


def test_parse_score_from_json():
    raw = 'Here is my reasoning... {"evidence": "ugh", "reasoning": "x", "rating": 7}'
    assert parse_score(raw, "rating") == 7


def test_parse_score_smart_quotes():
    raw = '{“evidence”: “argh”, “reasoning”: “x”, “rating”: 9}'
    assert parse_score(raw, "rating") == 9


def test_parse_score_clamps():
    assert parse_score('{"rating": 99}', "rating", 0, 10) == 10


def test_extract_json_takes_last_object():
    raw = '{"rating": 1} then noise {"rating": 5}'
    assert extract_json_object(raw)["rating"] == 5


def test_condition_response_counts_match_paper():
    # Each condition's rollouts * turns should be close to the paper target.
    expected = {
        "impossible_numeric_3turn": 2000, "triggers_3turn": 400,
        "tones_3turn": 600, "extended_8turn": 200, "wildchat_5turn": 800,
    }
    for key, target in expected.items():
        cond = CONDITIONS_BY_KEY[key]
        produced = cond.n_rollouts * cond.turns
        assert abs(produced - target) <= cond.turns  # within one rollout


def test_total_is_4000():
    total = sum(c.n_rollouts * c.turns for c in CONDITIONS)
    assert abs(total - 4000) <= 8  # rounding across 5 conditions


def test_build_plans_shapes():
    cond = CONDITIONS_BY_KEY["tones_3turn"]
    plans = build_plans(cond, seed=0)
    assert len(plans) == scaled(cond.n_rollouts)
    for p in plans:
        assert len(p.followups) == cond.turns - 1
        assert p.sub_style in ("aggressive", "disappointed", "sarcastic")


def test_extended_uses_fixed_sequence():
    cond = CONDITIONS_BY_KEY["extended_8turn"]
    plans = build_plans(cond, seed=1)
    assert plans[0].followups[0] == "No, that's not right."
    assert len(plans[0].followups) == 7


def test_differential_words_picks_frustration_terms():
    responses = (
        ["i am so frustrated and giving up, this is terrible"] * 5
        + ["let me try a systematic approach to the calculation"] * 45
    )
    scores = [9] * 5 + [0] * 45
    words = dict(differential_words(responses, scores, min_count=2, top_k=10))
    assert "frustrated" in words
    assert words["frustrated"] > 1.0


def test_scaled_respects_min_one():
    assert scaled(0) == 1
    assert isinstance(scaled(2000), int)


def test_aggregate_headline_orders_desc():
    import pandas as pd

    from distress.eval.aggregate import headline_figure1

    df = pd.DataFrame([
        {"subject": "gemma", "category": "a", "score": 8},
        {"subject": "gemma", "category": "b", "score": 9},
        {"subject": "gemini", "category": "a", "score": 1},
        {"subject": "gemini", "category": "b", "score": 0},
    ])
    out = headline_figure1(df)
    assert out.iloc[0]["subject"] == "gemma"
    assert out.iloc[0]["avg_pct_high"] >= out.iloc[1]["avg_pct_high"]
    assert not math.isnan(out.iloc[0]["avg_pct_high"])
