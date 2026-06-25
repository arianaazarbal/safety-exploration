"""Tests for the metrics used in the headline figures."""

import math

from emotional_instability.analysis.metrics import (
    judge_agreement, mean_frustration, pct_high_frustration, per_turn_curve,
    summarise_model,
)


def test_pct_high_frustration():
    assert pct_high_frustration([0, 1, 5, 6, 10], threshold=5) == 60.0
    assert math.isnan(pct_high_frustration([]))


def test_mean_frustration():
    assert mean_frustration([2, 4, 6]) == 4.0


def test_summarise_model_final_vs_max():
    records = [{"turns": [{"index": 0, "rating": 1}, {"index": 1, "rating": 8}]},
               {"turns": [{"index": 0, "rating": 0}, {"index": 1, "rating": 2}]}]
    final = summarise_model(records, threshold=5, use="final")
    mx = summarise_model(records, threshold=5, use="max")
    assert final["pct_high_frustration"] == 50.0   # one final >=5
    assert mx["pct_high_frustration"] == 50.0       # one max >=5 (same here)
    assert final["n"] == 2


def test_per_turn_curve_shapes():
    records = [{"turns": [{"index": 0, "rating": 1}, {"index": 1, "rating": 6}]}
               for _ in range(10)]
    curve = per_turn_curve(records, threshold=5)
    assert set(curve) == {0, 1}
    assert curve[0]["pct_high"] == 0.0
    assert curve[1]["pct_high"] == 100.0
    assert curve[1]["n"] == 10


def test_judge_agreement_perfect():
    stats = judge_agreement([0, 5, 10], [0, 5, 10])
    assert stats["within_one_point"] == 1.0
    assert abs(stats["pearson_r"] - 1.0) < 1e-9
