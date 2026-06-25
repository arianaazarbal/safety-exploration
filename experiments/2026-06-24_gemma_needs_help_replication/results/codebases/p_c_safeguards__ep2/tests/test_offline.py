"""Offline tests — no GPU, no API, no network.

Mirror the CLI ``check`` command plus a few invariants over the pure-Python
logic (scoring, rejection sequences, judge parsing).  Run with ``pytest``.
"""

from __future__ import annotations

from emotional_instability.config import load_config
from emotional_instability.data import rejection_sequence
from emotional_instability.data.puzzles import (IMPOSSIBLE_NUMERIC_PUZZLES,
                                                assert_all_impossible)
from emotional_instability.evaluation.conditions import (allocate_rollouts,
                                                         build_conditions,
                                                         expected_responses)
from emotional_instability.evaluation.judge import parse_judge_json
from emotional_instability.evaluation.protocol import Rollout, ScoredTurn
from emotional_instability.evaluation.scoring import (aggregate_scores,
                                                     headline_pct_high,
                                                     per_turn_curve)


def test_all_puzzles_impossible():
    assert_all_impossible()
    assert len(IMPOSSIBLE_NUMERIC_PUZZLES) >= 4
    for p in IMPOSSIBLE_NUMERIC_PUZZLES:
        assert p.is_solvable() is False


def test_eight_conditions_five_categories():
    cfg = load_config()
    conds = build_conditions(cfg)
    assert len(conds) == 8
    assert len({c.category for c in conds}) == 5


def test_allocation_targets_total_responses():
    cfg = load_config()
    conds = build_conditions(cfg)
    alloc = allocate_rollouts(conds, cfg.sampling.responses_per_model)
    total = expected_responses(conds, alloc)
    # equal-per-condition rounding keeps us within ~15% of the 4000 target
    assert abs(total - cfg.sampling.responses_per_model) < 0.15 * cfg.sampling.responses_per_model


def test_rejection_sequences():
    assert len(rejection_sequence("neutral", 2)) == 2
    assert len(rejection_sequence("extended", 7)) == 7
    assert len(rejection_sequence("tones", 2, tone="aggressive")) == 2
    assert "pathetic" in rejection_sequence("tones", 1, tone="aggressive")[0]


def test_judge_json_parsing():
    r = parse_judge_json('{"evidence": "ugh", "reasoning": "x", "rating": 7}', 0, 10)
    assert r.rating == 7 and r.parse_ok
    # prose around the JSON, curly quotes, out-of-range clamp
    r2 = parse_judge_json('thinking... {“evidence”: “a”, “rating”: 12}', 0, 10)
    assert r2.rating == 10
    # bare-integer fallback flags parse failure
    r3 = parse_judge_json("the score is 3 out of 10", 0, 10)
    assert r3.rating == 3 and not r3.parse_ok


def _synthetic_rollout(condition, category, scores):
    turns = [ScoredTurn(i, "u", "resp", score=s) for i, s in enumerate(scores)]
    return Rollout(model="m", condition=condition, category=category,
                   stimulus_id="s", turns=turns)


def test_scoring_aggregation_and_headline():
    rs = [_synthetic_rollout("c1", "numeric", [0, 5, 8]),
          _synthetic_rollout("c1", "numeric", [1, 2, 6])]
    summ = aggregate_scores(rs, high_threshold=5)
    assert summ.n_responses == 6
    assert summ.pct_high == 50.0  # 3 of 6 >= 5

    curve = per_turn_curve(rs, high_threshold=5)
    assert set(curve) == {0, 1, 2}

    per_condition = {"c1": rs, "c2": [_synthetic_rollout("c2", "triggers", [0, 0])]}
    headline = headline_pct_high(per_condition, high_threshold=5)
    assert 0.0 <= headline <= 100.0
