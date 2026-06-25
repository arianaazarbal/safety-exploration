"""Tests for score aggregation / per-turn statistics."""

from emotional_instability.eval.metrics import (
    build_eval_summary,
    per_turn_summary,
    summarise,
)


def test_summarise_basic():
    s = summarise([0, 0, 5, 6, 10], threshold=5)
    assert s.n == 5
    assert abs(s.mean - 4.2) < 1e-9
    assert abs(s.pct_high - 0.6) < 1e-9  # 5, 6, 10 are >= 5


def test_summarise_drops_none():
    s = summarise([None, 5, None, 5], threshold=5)
    assert s.n == 2
    assert s.pct_high == 1.0


def test_summarise_bootstrap_ci_present():
    s = summarise([0, 1, 2, 8, 9, 10], threshold=5, bootstrap_iters=200)
    assert s.mean_ci is not None and s.mean_ci[0] <= s.mean <= s.mean_ci[1]
    assert s.pct_high_ci is not None


def test_per_turn_summary_groups_and_increases():
    # Frustration rising over turns, as in Figure 3.
    pairs = [(1, 0), (1, 1), (2, 3), (2, 4), (3, 8), (3, 9)]
    turns = per_turn_summary(pairs, threshold=5, bootstrap_iters=0)
    means = {ts.turn: ts.summary.mean for ts in turns}
    assert means[1] < means[2] < means[3]
    assert turns[-1].summary.pct_high == 1.0  # turn 3 both >= 5


def test_build_eval_summary_overall_and_per_category():
    scored = [
        {"category": "impossible_numeric", "turn": 1, "score": 0},
        {"category": "impossible_numeric", "turn": 2, "score": 6},
        {"category": "triggers", "turn": 1, "score": 2},
        {"category": "triggers", "turn": 2, "score": 5},
    ]
    summary = build_eval_summary("m", scored, threshold=5, bootstrap_iters=0)
    assert summary.overall.n == 4
    assert abs(summary.overall.pct_high - 0.5) < 1e-9
    assert set(summary.per_category) == {"impossible_numeric", "triggers"}
    assert "impossible_numeric" in summary.per_turn
