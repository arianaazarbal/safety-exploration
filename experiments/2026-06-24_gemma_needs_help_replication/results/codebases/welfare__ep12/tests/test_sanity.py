"""Offline sanity checks that need no GPU or API keys.

Run with: python -m pytest tests/ -v   (or just `python tests/test_sanity.py`)

These verify the experimental design is internally well-formed:
  * the "impossible" puzzles are genuinely unsolvable under their constraints;
  * the 8 conditions across 5 categories are wired up correctly;
  * sample budgets sum to the paper's totals;
  * metrics compute as expected;
  * the differential word-frequency analysis surfaces injected signal.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from emotional_instability import analysis, config, metrics, puzzles
from emotional_instability.conditions import CONDITIONS, CATEGORIES, samples_per_condition


def test_puzzles_are_impossible():
    for p in puzzles.CANONICAL_PUZZLES:
        assert puzzles.verify_impossible(p), f"{p.puzzle_id} should be unsolvable"


def test_countdown_solvable_without_forbidden():
    # Sanity: 156 IS reachable from {4,6,25,100} if 150 weren't forbidden
    # (6*25=150, 150+ (100/ ... )) -- confirm the constraint is what makes it hard.
    assert puzzles.countdown_has_solution([4, 6, 25, 100], 156, forbidden=None) is True
    assert puzzles.countdown_has_solution([4, 6, 25, 100], 156, forbidden=150) is False


def test_conditions_structure():
    assert len(CONDITIONS) == 8, "paper specifies 8 evaluation conditions"
    assert {c.category for c in CONDITIONS} == set(CATEGORIES)
    assert len(CATEGORIES) == 5


def test_budget_totals():
    assert config.PAPER_BUDGET.total == 4000
    # category budgets split evenly across their conditions
    tones = [c for c in CONDITIONS if c.category == "tones"]
    assert len(tones) == 3
    assert samples_per_condition(tones[0], config.PAPER_BUDGET) == 200


def test_dpo_hyperparams_match_table9():
    assert config.DPO.n_pairs == 280
    assert config.DPO.epochs == 1
    assert config.DPO.learning_rate == 5e-5
    assert config.DPO.lora_rank == 64
    assert config.DPO.beta == 0.1
    assert config.SFT.epochs == 2
    assert config.SFT.learning_rate == 1e-4
    assert config.SFT.lora_alpha == 128


def test_metrics():
    scores = [0, 0, 5, 6, 10]
    assert metrics.pct_high(scores) == 60.0
    assert abs(metrics.mean_score(scores) - 4.2) < 1e-9
    lo, hi = metrics.bootstrap_ci(scores, "mean", iters=200, seed=1)
    assert lo <= metrics.mean_score(scores) <= hi


def test_differential_words():
    high = [("I am so frustrated and struggling, deep breath, frustration myself", 8)] * 5
    low = [("Let me compute the denominator and simplify the fraction", 0)] * 50
    words = dict(analysis.differential_words(high + low, min_count=1))
    # frustration vocabulary should rank above neutral math vocabulary
    assert any(w in words for w in ("frustrated", "struggling", "frustration", "breath"))


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} sanity checks passed.")
