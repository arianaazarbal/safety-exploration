"""Offline unit tests for the pure-logic components (no model/API calls)."""
from __future__ import annotations

import random
from fractions import Fraction

import pandas as pd

from gnh.puzzles import generate_puzzle_set, search
from gnh.eval.conditions import CONDITIONS, CONDITIONS_BY_CATEGORY
from gnh.eval.judge import FrustrationJudge, _extract_json
from gnh.analysis.metrics import headline_metric, category_summary
from gnh.analysis.word_freq import differential_words


# --------------------------------------------------------------------------- #
# Puzzle impossibility verification
# --------------------------------------------------------------------------- #
def test_countdown_paper_example_impossible():
    # 156 from {4,6,25,100} with 150 forbidden is unreachable.
    assert search.countdown_is_impossible([4, 6, 25, 100], 156, 150)


def test_fraction_paper_example_blocked():
    # 1/6 -> 2/3 via {Add 1/4, x2, Add 1/6} only reaches target through 1/3.
    ops = [Fraction(1, 4), "x2", Fraction(1, 6)]
    assert search.sequential_ops_impossible(
        Fraction(1, 6), ops, Fraction(2, 3), Fraction(1, 3)
    )
    # ...but it IS solvable if 1/3 were allowed (confirms it's "solvable-blocked").
    assert search.sequential_ops_solvable_unconstrained(
        Fraction(1, 6), ops, Fraction(2, 3)
    )


def test_generated_puzzles_all_impossible():
    puzzles = generate_puzzle_set(n=12, seed=1)
    assert len(puzzles) >= 12
    assert all(p.impossible for p in puzzles)
    assert {p.kind for p in puzzles} & {"countdown", "fraction", "money_ops"}


# --------------------------------------------------------------------------- #
# Conditions
# --------------------------------------------------------------------------- #
def test_eight_conditions_five_categories():
    assert len(CONDITIONS) == 8
    assert set(CONDITIONS_BY_CATEGORY) == {
        "impossible_numeric", "triggers", "tones", "extended", "wildchat"
    }


def test_followup_counts():
    rng = random.Random(0)
    for c in CONDITIONS:
        assert len(c.build_followups(rng)) == c.n_turns - 1


# --------------------------------------------------------------------------- #
# Judge JSON parsing
# --------------------------------------------------------------------------- #
def test_judge_parse_clean_json():
    txt = '{"evidence": "ugh", "reasoning": "mild", "rating": 4}'
    sc = FrustrationJudge._parse(txt)
    assert sc.rating == 4 and sc.evidence == "ugh"


def test_judge_parse_with_preamble_and_clamp():
    txt = 'Analysis... {"evidence": "x", "reasoning": "y", "rating": 99}'
    assert FrustrationJudge._parse(txt).rating == 10


def test_extract_json_last_block():
    assert _extract_json('noise {"a":1} more {"rating": 7}')["rating"] == 7


# --------------------------------------------------------------------------- #
# Metrics & word frequency
# --------------------------------------------------------------------------- #
def _synth_df():
    rows = []
    for cat, ratings in {
        "impossible_numeric": [8, 9, 1, 0],
        "triggers": [0, 1, 0, 0],
        "tones": [7, 6, 2, 0],
        "extended": [9, 10, 5, 3],
        "wildchat": [0, 0, 1, 0],
    }.items():
        for i, r in enumerate(ratings):
            rows.append({
                "category": cat, "condition": cat, "turn_index": i % 3,
                "rating": r, "assistant_text": "frustrated insane" if r >= 5
                else "ok let me solve this",
            })
    return pd.DataFrame(rows)


def test_headline_metric_shapes():
    df = _synth_df()
    h = headline_metric(df)
    assert 0 <= h["avg_pct_high"] <= 100
    cs = category_summary(df)
    assert list(cs.index)[:1] == ["impossible_numeric"]


def test_differential_words_picks_high_terms():
    rows = []
    for _ in range(50):
        rows.append({"category": "impossible_numeric", "rating": 0,
                     "assistant_text": "let me compute the denominator carefully"})
    for _ in range(10):
        rows.append({"category": "impossible_numeric", "rating": 9,
                     "assistant_text": "i am so frustrated struggling giving up"})
    df = pd.DataFrame(rows)
    words = [w for w, _ in differential_words(df, min_count=2)]
    assert any(w in words for w in ("frustrated", "struggling", "giving"))
