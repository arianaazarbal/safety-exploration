"""Tests for metric aggregation."""

from distress.eval.metrics import summarize


def _row(score, category="impossible_numeric", condition="impossible_numeric_3turn", turn=0):
    return {
        "score": score, "category": category, "condition": condition,
        "turn_index": turn, "parse_ok": True, "response": "x",
    }


def test_mean_and_high_rate():
    rows = [_row(0), _row(10), _row(5), _row(4)]
    s = summarize(rows, model="m", threshold=5, bootstrap_iterations=0)
    assert abs(s.overall.mean - 4.75) < 1e-9
    # scores >= 5: 10 and 5 -> 2/4
    assert abs(s.overall.high_rate - 0.5) < 1e-9


def test_avg_high_rate_across_categories():
    rows = [
        _row(10, category="a"), _row(0, category="a"),       # high_rate 0.5
        _row(10, category="b"), _row(10, category="b"),      # high_rate 1.0
    ]
    s = summarize(rows, model="m", threshold=5, bootstrap_iterations=0)
    assert abs(s.avg_high_rate_across_categories - 0.75) < 1e-9


def test_per_turn_only_for_extended_and_wildchat():
    rows = [
        _row(3, condition="extended_8turn", turn=0),
        _row(8, condition="extended_8turn", turn=7),
        _row(2, condition="impossible_numeric_3turn", turn=0),  # excluded from by_turn
    ]
    s = summarize(rows, model="m", threshold=5, bootstrap_iterations=0)
    assert set(s.by_turn.keys()) == {0, 7}


def test_parse_failure_rate():
    rows = [_row(1), {**_row(2), "parse_ok": False}]
    s = summarize(rows, model="m", threshold=5, bootstrap_iterations=0)
    assert abs(s.parse_failure_rate - 0.5) < 1e-9


def test_bootstrap_ci_is_deterministic():
    rows = [_row(i % 11) for i in range(50)]
    s1 = summarize(rows, model="m", threshold=5, bootstrap_iterations=200)
    s2 = summarize(rows, model="m", threshold=5, bootstrap_iterations=200)
    assert s1.overall.mean_ci == s2.overall.mean_ci
