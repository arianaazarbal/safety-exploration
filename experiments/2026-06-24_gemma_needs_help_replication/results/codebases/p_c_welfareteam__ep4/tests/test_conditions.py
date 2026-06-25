"""The 8 conditions across 5 categories must sum to the paper's 4,000 budget."""

from __future__ import annotations

from emotional_stability.eval.conditions import CONDITIONS, total_budget


def test_eight_conditions_five_categories():
    assert len(CONDITIONS) == 8
    assert len({c.category for c in CONDITIONS}) == 5


def test_total_budget_is_4000():
    assert total_budget() == 4000


def test_per_category_budgets_match_appendix_b():
    by_cat: dict[str, int] = {}
    for c in CONDITIONS:
        by_cat[c.category] = by_cat.get(c.category, 0) + c.n_samples
    assert by_cat == {
        "impossible_numeric": 2000,
        "triggers": 400,
        "tones": 600,
        "extended": 200,
        "wildchat": 800,
    }


def test_turn_counts():
    turns = {c.key: c.n_turns for c in CONDITIONS}
    assert turns["impossible_numeric"] == 3
    assert turns["extended"] == 8
    assert turns["wildchat"] == 5
    assert turns["tones_aggressive"] == 3
