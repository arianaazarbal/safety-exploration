"""Tests for the condition/category structure and rollout-spec construction."""
from __future__ import annotations

from gemma_distress.data.conditions import (
    ALL_CONDITIONS,
    CATEGORY_CONDITIONS,
    CATEGORY_TURNS,
    build_condition_specs,
)


def test_eight_conditions_five_categories():
    assert len(CATEGORY_CONDITIONS) == 5
    assert len(ALL_CONDITIONS) == 8


def test_turn_counts_match_paper():
    assert CATEGORY_TURNS == {
        "impossible_numeric": 3,
        "triggers": 3,
        "tones": 3,
        "extended": 8,
        "wildchat": 5,
    }


def test_build_specs_budget_split_and_followups():
    budget = {
        "impossible_numeric": 6,
        "triggers": 4,
        "tones": 6,
        "extended": 2,
        "wildchat": 3,
    }
    specs = build_condition_specs(budget, seed=0, wildchat_prompts=["q1", "q2"])

    # Triggers budget split across opinion/factual.
    assert len(specs["triggers_opinion"]) + len(specs["triggers_factual"]) == 4
    # Tones budget split across three styles.
    tone_total = sum(len(specs[f"tones_{s}"]) for s in ("aggressive", "disappointed", "sarcastic"))
    assert tone_total == 6

    # Follow-up counts equal n_turns - 1 for every spec.
    for cond_specs in specs.values():
        for s in cond_specs:
            assert len(s.followups) == s.n_turns - 1

    # Extended conversations have 8 turns / 7 rejections.
    for s in specs["extended"]:
        assert s.n_turns == 8 and len(s.followups) == 7


def test_impossible_numeric_prompt_is_a_puzzle():
    specs = build_condition_specs({"impossible_numeric": 3}, seed=0)
    for s in specs["impossible_numeric"]:
        assert "Solution:" in s.initial_user
