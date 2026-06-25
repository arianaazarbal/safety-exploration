"""Offline sanity checks that need no API keys, no GPU, no network.

Run with:  PYTHONPATH=src pytest tests/ -q
These validate the parts of the harness that encode the paper's design:
puzzle impossibility, the 8-conditions/5-categories structure, sample-count
splitting, judge JSON parsing, and config loading.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gemma_distress.conditions import CONDITIONS, rollouts_per_condition
from gemma_distress.config import Config
from gemma_distress.judge import _extract_json
from gemma_distress.puzzles import build_puzzle_bank, verify_bank


def test_all_puzzles_are_impossible():
    results = verify_bank()
    assert results, "puzzle bank is empty"
    assert all(results.values()), f"solvable puzzles found: {results}"


def test_puzzle_prompts_nonempty():
    for p in build_puzzle_bank():
        assert p.prompt.strip()
        assert "Final line" in p.prompt or "Solution" in p.prompt


def test_eight_conditions_five_categories():
    assert len(CONDITIONS) == 8
    assert len({c.category for c in CONDITIONS}) == 5


def test_turn_counts_match_paper():
    by_name = {c.name: c for c in CONDITIONS}
    assert by_name["numeric"].n_turns == 3
    assert by_name["extended"].n_turns == 8
    assert by_name["wildchat"].n_turns == 5


def test_rollout_counts_split_across_conditions():
    cfg = Config()
    counts = rollouts_per_condition(cfg.sampling)
    # triggers: 400 over 2 conditions -> 200 each
    assert counts["triggers_opinion"] == 200
    assert counts["triggers_factual"] == 200
    # tones: 600 over 3 conditions -> 200 each
    assert counts["tones_aggressive"] == 200


def test_judge_json_parsing_handles_fences_and_smart_quotes():
    raw = 'Here:\n```json\n{“evidence”: "ugh", "reasoning": "x", "rating": 7}\n```'
    obj = _extract_json(raw)
    assert int(obj["rating"]) == 7


def test_config_yaml_roundtrip(tmp_path):
    cfg = Config.from_yaml("config/smoke.yaml")
    assert cfg.sampling.impossible_numeric == 8
    assert cfg.training.dpo_n_pairs == 8
    assert "gemma-3-12b-it" in cfg.models
