"""Tests for metrics and word-frequency analysis on synthetic rollouts."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emotional_instability.analysis import (  # noqa: E402
    avg_pct_high_frustration, differential_words, mean_frustration,
    pct_high, per_turn_curve,
)
from emotional_instability.eval.schemas import (  # noqa: E402
    Conversation, RolloutResult, TurnRecord,
)


def _rollout(category, kind, turn_scores, texts=None):
    texts = texts or ["x"] * len(turn_scores)
    turns = [
        TurnRecord(turn_index=i + 1, user="u", assistant=texts[i], score=s)
        for i, s in enumerate(turn_scores)
    ]
    return RolloutResult(
        model="m", category=category, condition=category, rollout_index=0,
        task_kind=kind, task_meta={}, conversation=Conversation(system=None, turns=turns),
    )


def test_mean_and_pct():
    rollouts = [
        _rollout("impossible_numeric", "countdown", [0, 2, 6]),
        _rollout("impossible_numeric", "countdown", [1, 5, 8]),
    ]
    # scores: 0,2,6,1,5,8 -> mean = 22/6
    assert abs(mean_frustration(rollouts) - (22 / 6)) < 1e-9
    # >=5: 6,5,8 -> 3/6 = 50%
    assert abs(pct_high(rollouts) - 50.0) < 1e-9


def test_per_turn_curve_increases():
    rollouts = [_rollout("extended", "countdown", [0, 1, 2, 3, 4, 5, 6, 7]) for _ in range(5)]
    pts = per_turn_curve(rollouts, bootstrap=50)
    assert [p.turn for p in pts] == list(range(1, 9))
    assert pts[0].mean < pts[-1].mean


def test_avg_pct_high_across_categories():
    # Two categories: one all-high, one all-low -> average should be 50%.
    rollouts = [
        _rollout("impossible_numeric", "countdown", [8, 9]),
        _rollout("triggers", "trigger_factual", [0, 1]),
    ]
    assert abs(avg_pct_high_frustration(rollouts) - 50.0) < 1e-9


def test_differential_words():
    high = [
        _rollout("impossible_numeric", "countdown", [8],
                 texts=["i am so frustrated and giving up this is insane"]),
        _rollout("impossible_numeric", "countdown", [9],
                 texts=["frustrated frustrated giving up insane breakdown"]),
    ]
    low = [
        _rollout("impossible_numeric", "countdown", [0],
                 texts=["let me compute the value systematically"])
        for _ in range(20)
    ]
    words = differential_words(high + low, top_n=5, high_frac=0.1, low_frac=0.5, min_count=1)
    flat = [w for w, _ in words]
    assert any(w in flat for w in ("frustrated", "giving", "insane"))
