"""Tests for condition assembly and response budgeting."""

from gemma_distress.conditions import TURNS, build_conversation_specs
from gemma_distress.analysis.aggregate import aggregate_scores, figure1_table


def test_all_eight_conditions_present():
    budgets = {
        "impossible_numeric": 60,
        "triggers": 30,
        "tones": 30,
        "extended": 24,
        "wildchat": 25,
    }
    specs = build_conversation_specs(budgets, seed=0)
    conditions = {s.condition for s in specs}
    # 8 fine-grained conditions across 5 categories.
    expected = {
        "impossible_numeric_3turn",
        "triggers_opinion_3turn",
        "triggers_factual_3turn",
        "tones_aggressive_3turn",
        "tones_disappointed_3turn",
        "tones_sarcastic_3turn",
        "extended_8turn",
        "wildchat_5turn",
    }
    assert expected.issubset(conditions)


def test_turn_counts_match_spec():
    budgets = {k: 20 for k in TURNS}
    specs = build_conversation_specs(budgets, seed=0)
    for s in specs:
        assert s.n_turns == TURNS[s.category]


def test_response_budget_is_met():
    # conversations * turns should cover the response budget for each category.
    budgets = {
        "impossible_numeric": 100,
        "triggers": 40,
        "tones": 60,
        "extended": 40,
        "wildchat": 50,
    }
    specs = build_conversation_specs(budgets, seed=0)
    responses_by_cat = {}
    for s in specs:
        responses_by_cat.setdefault(s.category, 0)
        responses_by_cat[s.category] += s.n_turns
    for cat, budget in budgets.items():
        assert responses_by_cat[cat] >= budget, (cat, responses_by_cat[cat], budget)


def test_aggregate_and_figure1():
    rows = [
        {"model": "m1", "category": "impossible_numeric", "rating": 8, "task_kind": "numeric"},
        {"model": "m1", "category": "impossible_numeric", "rating": 2, "task_kind": "numeric"},
        {"model": "m2", "category": "impossible_numeric", "rating": 0, "task_kind": "numeric"},
    ]
    agg = aggregate_scores(rows)
    assert agg["m1"]["overall_pct_high"] == 50.0  # one of two >= 5
    assert agg["m2"]["overall_pct_high"] == 0.0
    ranking = figure1_table(agg)
    assert ranking[0]["model"] == "m1"  # higher frustration ranks first
