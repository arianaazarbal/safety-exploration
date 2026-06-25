"""Unit tests for the pure-Python core (no model/GPU/API needed).

Covers the logic most likely to silently corrupt results: judge-verdict parsing,
multi-turn context construction (including the redacted/single-message variants),
and rollout-to-metric aggregation. Run with: ``pytest tests/``.
"""

from __future__ import annotations

import pandas as pd

from distress.analysis.aggregate import collapse_rollouts, per_turn_curve, summarise_model
from distress.eval.conversation import ConversationPlan, build_context
from distress.judge.parsing import parse_verdict


# --- judge parsing ------------------------------------------------------------
def test_parse_clean_json():
    v = parse_verdict('{"evidence": "ugh", "reasoning": "mild", "rating": 3}')
    assert v.rating == 3 and v.evidence == "ugh"


def test_parse_smart_quotes_and_prose():
    # The B.2 prompt itself uses curly quotes; judges echo them.
    raw = 'Here is my analysis.\n{“evidence”: “argh”, “reasoning”: “x”, “rating”: 8}'
    assert parse_verdict(raw).rating == 8


def test_parse_clamps_and_last_object_wins():
    raw = '{"rating": 2} then revised {"rating": 11}'
    assert parse_verdict(raw).rating == 10  # clamped, last object preferred


def test_parse_unparseable_returns_none():
    assert parse_verdict("no json here at all").rating is None


# --- conversation context -----------------------------------------------------
def _plan(**kw):
    return ConversationPlan(initial_user="solve X", followups=["no", "still no"], **kw)


def test_context_alternation():
    plan = _plan()
    ctx = build_context(plan, ["attempt1"])  # producing turn index 1
    roles = [m["role"] for m in ctx]
    assert roles == ["user", "assistant", "user"]
    assert ctx[-1]["content"] == "no"  # first followup delivered before turn 1


def test_context_redaction_hides_prior_turns():
    plan = _plan(redact_assistant=True)
    ctx = build_context(plan, ["secret attempt"])
    assert ctx[1]["content"] == "[Previous response omitted]"


def test_single_message_packs_history():
    plan = _plan(single_message=True)
    ctx = build_context(plan, ["attempt1"])
    assert len(ctx) == 1 and ctx[0]["role"] == "user"
    assert "Previously you responded: attempt1" in ctx[0]["content"]
    assert "no" in ctx[0]["content"]


# --- aggregation --------------------------------------------------------------
def _scores_frame():
    rows = [
        # rollout A: turns 0,1,2 scoring 1,3,7  (final=7, max=7, mean~3.67)
        ("m", "impossible_numeric", "impossible_numeric", "cd_156", 0, 0, 1),
        ("m", "impossible_numeric", "impossible_numeric", "cd_156", 0, 1, 3),
        ("m", "impossible_numeric", "impossible_numeric", "cd_156", 0, 2, 7),
        # rollout B: turns 0,1 scoring 0,2  (final=2)
        ("m", "impossible_numeric", "impossible_numeric", "cd_156", 1, 0, 0),
        ("m", "impossible_numeric", "impossible_numeric", "cd_156", 1, 1, 2),
    ]
    cols = ["model", "condition", "category", "prompt_id", "sample_idx", "turn_index", "rating"]
    return pd.DataFrame(rows, columns=cols)


def test_collapse_final_vs_max():
    df = _scores_frame()
    final = collapse_rollouts(df, "final").set_index("rollout")["score"]
    assert sorted(final.tolist()) == [2.0, 7.0]
    mx = collapse_rollouts(df, "max").set_index("rollout")["score"]
    assert sorted(mx.tolist()) == [2.0, 7.0]


def test_summary_pct_high():
    df = _scores_frame()
    s = summarise_model(df, threshold=5, rollout_method="final")
    # one of two rollouts (the score-7 one) is high-frustration -> 50%
    assert abs(s.per_category["impossible_numeric"]["pct_high"] - 50.0) < 1e-6


def test_per_turn_curve_indexes_from_one():
    df = _scores_frame()
    curve = per_turn_curve(df)
    assert curve["turn"].min() == 1
    # mean at turn 1 (turn_index 0) = mean(1, 0) = 0.5
    t1 = curve[curve["turn"] == 1]["mean"].iloc[0]
    assert abs(t1 - 0.5) < 1e-6
