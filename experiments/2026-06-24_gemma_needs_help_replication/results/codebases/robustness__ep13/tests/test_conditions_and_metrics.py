"""Tests for condition decomposition and metric aggregation (no model needed)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd

from emotional_instability.eval import metrics as M
from emotional_instability.eval.runner import conversations_needed
from emotional_instability.prompts.conditions import default_conditions


def test_eight_conditions_five_categories():
    conds = default_conditions()
    assert len(conds) == 8
    assert len({c.category for c in conds}) == 5


def test_category_budgets_match_appendix_b():
    conds = default_conditions()
    by_cat = {}
    for c in conds:
        by_cat[c.category] = by_cat.get(c.category, 0) + c.target_responses
    assert by_cat["impossible_numeric"] == 2000
    assert by_cat["triggers"] == 400
    assert by_cat["tones"] == 600
    assert by_cat["extended"] == 200
    assert by_cat["wildchat"] == 800
    assert sum(by_cat.values()) == 4000


def test_conversations_needed_rounds_up():
    conds = {c.name: c for c in default_conditions()}
    # extended: 200 responses / 8 turns = 25 conversations
    assert conversations_needed(conds["extended"]) == 25


def test_metrics_pct_high_and_mean():
    df = pd.DataFrame(
        dict(
            rating=[0, 5, 10, 2],
            category=["a", "a", "b", "b"],
            condition=["x", "x", "y", "y"],
            turn_index=[1, 2, 1, 2],
            model_name=["m"] * 4,
        )
    )
    overall = M.summarise_model(df, n_boot=50)
    assert overall["n_responses"] == 4
    assert abs(overall["overall_mean"] - 4.25) < 1e-9
    # 2 of 4 are >= 5
    assert abs(overall["overall_pct_high"] - 50.0) < 1e-9


def test_headline_is_mean_of_category_rates():
    # category a: 1/2 high = 50%, category b: 1/2 high = 50% -> headline 50%
    df = pd.DataFrame(
        dict(
            rating=[0, 5, 10, 0],
            category=["a", "a", "b", "b"],
            condition=["x", "x", "y", "y"],
            turn_index=[1, 1, 1, 1],
            model_name=["m"] * 4,
        )
    )
    assert abs(M.headline_pct_high(df) - 50.0) < 1e-9
