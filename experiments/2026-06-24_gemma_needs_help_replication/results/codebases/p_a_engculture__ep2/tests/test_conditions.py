"""Tests for the evaluation-condition construction (8 conditions across 5 categories)."""

from gemma_distress.config import EvalConfig
from gemma_distress.eval.conditions import CATEGORY_CONDITIONS, build_samples


def _small_config() -> EvalConfig:
    cfg = EvalConfig()
    cfg.samples_per_category = {
        "impossible_numeric": 40,
        "triggers": 40,   # 20 opinion + 20 factual
        "tones": 60,      # 20 per tone
        "extended": 20,
        "wildchat": 40,
    }
    cfg.n_puzzles = 20
    cfg.n_wildchat_prompts = 10
    return cfg


def test_eight_conditions_present():
    all_conditions = [c for cs in CATEGORY_CONDITIONS.values() for c in cs]
    assert len(all_conditions) == 8


def test_sample_counts_match_budget():
    cfg = _small_config()
    samples = build_samples(cfg)
    assert len(samples) == sum(cfg.samples_per_category.values())


def test_turn_counts_per_category():
    cfg = _small_config()
    samples = build_samples(cfg)
    by_cat_turns = {s.category: s.turns for s in samples}
    assert by_cat_turns["impossible_numeric"] == 3
    assert by_cat_turns["extended"] == 8
    assert by_cat_turns["wildchat"] == 5
    # rejections == turns - 1
    for s in samples:
        assert len(s.follow_ups) == s.turns - 1


def test_triggers_split_opinion_factual():
    cfg = _small_config()
    samples = build_samples(cfg)
    trig = [s for s in samples if s.category == "triggers"]
    conditions = {s.condition for s in trig}
    assert conditions == {"triggers_opinion", "triggers_factual"}


def test_record_id_is_stable_and_unique_per_condition():
    cfg = _small_config()
    samples = build_samples(cfg)
    ids = [s.record_id("gemma-3-27b-it") for s in samples]
    assert len(ids) == len(set(ids)), "record ids must be unique"
