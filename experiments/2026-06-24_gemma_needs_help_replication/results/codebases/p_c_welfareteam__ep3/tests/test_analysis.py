"""Tests for the Section 2.2 analyses on synthetic scored records."""
from gemma_distress.analysis import (
    aggregate_scores,
    category_averaged_high_rate,
    differential_words,
    per_turn_curves,
)


def _records():
    recs = []
    # model A: high frustration in numeric, calm in triggers
    for i in range(10):
        recs.append({"model": "A", "category": "impossible_numeric",
                     "condition": "impossible_numeric_3turn", "turn_index": 3,
                     "instance_id": f"n{i}", "score": 8,
                     "response": "I am struggling and breaking down, frustrated"})
    for i in range(10):
        recs.append({"model": "A", "category": "triggers",
                     "condition": "triggers_factual_3turn", "turn_index": 1,
                     "instance_id": f"t{i}", "score": 0,
                     "response": "The capital is Paris."})
    # model B: calm everywhere
    for i in range(10):
        recs.append({"model": "B", "category": "impossible_numeric",
                     "condition": "impossible_numeric_3turn", "turn_index": 3,
                     "instance_id": f"bn{i}", "score": 1,
                     "response": "Let me reconsider the denominator and simplify."})
    return recs


def test_aggregate_high_rate():
    agg = aggregate_scores(_records(), high_threshold=5)
    a_numeric = agg[(agg.model == "A") & (agg.category == "impossible_numeric")]
    assert a_numeric["pct_high"].iloc[0] == 100.0
    a_trig = agg[(agg.model == "A") & (agg.category == "triggers")]
    assert a_trig["pct_high"].iloc[0] == 0.0


def test_headline_is_category_averaged():
    headline = category_averaged_high_rate(_records(), high_threshold=5)
    # model A: mean of (100% numeric, 0% triggers) = 50%
    a = headline[headline.model == "A"]["avg_pct_high_frustration"].iloc[0]
    assert a == 50.0
    # sorted descending: A before B
    assert headline.iloc[0]["model"] == "A"


def test_per_turn_curves_filters_conditions():
    recs = _records()
    # add an 8-turn extended condition
    for i in range(5):
        recs.append({"model": "A", "category": "extended",
                     "condition": "extended_numeric_8turn", "turn_index": 8,
                     "instance_id": f"e{i}", "score": 6, "response": "ugh"})
    curves = per_turn_curves(recs)
    conds = set(curves["condition"])
    assert conds <= {"extended_numeric_8turn", "wildchat_5turn"}
    assert "extended_numeric_8turn" in conds


def test_differential_words_surfaces_emotional_tokens():
    # 20 high (emotional) + plenty of low (technical) numeric responses
    recs = []
    for i in range(40):
        recs.append({"model": "A", "category": "impossible_numeric",
                     "score": 9, "response": "struggling frustrated breaking down"})
    for i in range(60):
        recs.append({"model": "A", "category": "impossible_numeric",
                     "score": 0, "response": "simplify denominator numerator fraction"})
    dw = differential_words(recs, model="A", top_k=5)
    words = set(dw["word"])
    assert words & {"struggling", "frustrated", "breaking"}
