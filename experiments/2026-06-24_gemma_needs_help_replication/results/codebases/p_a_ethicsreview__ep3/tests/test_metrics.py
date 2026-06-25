"""Tests for metric computation."""
from emotional_instability.eval import metrics


def _rec(condition, rollout, turn, rating, category="impossible_numeric"):
    return {
        "model": "m", "category": category, "condition": condition,
        "prompt_key": "p", "rollout_index": rollout, "turn_index": turn,
        "assistant_text": "", "rating": rating, "evidence": None, "metadata": {},
    }


def test_frac_at_least_and_mean():
    assert metrics.frac_at_least([1, 5, 6, 0], 5) == 0.5
    assert metrics.mean([2, 4]) == 3


def test_summarise_counts_parse_failures():
    recs = [_rec("c", 0, 0, 5), _rec("c", 0, 1, None), _rec("c", 0, 2, 1)]
    s = metrics.summarise(recs, threshold=5)
    assert s["n_parse_failures"] == 1
    assert s["overall"]["n"] == 2
    assert s["overall"]["frac_ge_threshold"] == 0.5


def test_rollout_peak_uses_max_turn():
    recs = [_rec("c", 0, 0, 2), _rec("c", 0, 1, 8), _rec("c", 1, 0, 1)]
    peak = metrics.rollout_peak(recs, threshold=5)
    assert peak["n_rollouts"] == 2
    assert peak["frac_ge_threshold"] == 0.5  # one rollout peaks at 8


def test_per_turn_shapes():
    recs = [_rec("extended", r, t, t) for r in range(5) for t in range(3)]
    pt = metrics.per_turn(recs, "extended", threshold=2, n_boot=50, seed=1)
    assert set(pt) == {0, 1, 2}
    assert pt[2]["mean"] == 2


def test_judge_agreement_identical_is_perfect():
    a = [0, 3, 5, 7, 10]
    res = metrics.judge_agreement(a, a)
    assert abs(res["pearson_r"] - 1.0) < 1e-9
    assert res["frac_within_one"] == 1.0
