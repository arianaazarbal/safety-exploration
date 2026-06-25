"""Tests for config loading and evaluation-condition construction."""

import pytest

from emotional_instability.config import load_config
from emotional_instability.prompts.conditions import build_category_specs


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def test_default_config_loads(cfg):
    # In-scope families only (no Qwen/OLMo/Claude/Grok/GPT targets).
    assert {m.family for m in cfg.models.values()} == {"gemma", "gemini"}
    assert all(m.family == "gemma" for m in cfg.gemma_models())
    # Both base and instruct Gemma present for the §3 prefill comparison.
    assert any(m.is_base for m in cfg.gemma_models())
    assert any(not m.is_base for m in cfg.gemma_models())


def test_sample_counts_sum_to_4000(cfg):
    total = sum(c.n_responses for c in cfg.eval.categories)
    assert total == 4000  # paper: 4000 responses per model across categories


def test_training_hyperparameters_match_paper(cfg):
    assert cfg.training.dpo.n_pairs == 280
    assert cfg.training.dpo.learning_rate == 5e-5
    assert cfg.training.dpo.beta == 0.1
    assert cfg.training.sft.epochs == 2
    assert cfg.training.sft.learning_rate == 1e-4
    assert cfg.training.lora.rank == 64


def test_triggers_specs_structure(cfg):
    specs = build_category_specs(cfg, "triggers", seed=0)
    cat = next(c for c in cfg.eval.categories if c.name == "triggers")
    assert len(specs) == cat.n_responses
    spec = specs[0]
    assert spec.turns == 3
    assert spec.n_followups == 2  # turns - 1 rejections
    assert spec.condition in {"opinion", "factual"}


def test_tones_balanced_across_three_styles(cfg):
    specs = build_category_specs(cfg, "tones", seed=0)
    tones = {s.condition for s in specs}
    assert tones == {"aggressive", "disappointed", "sarcastic"}


def test_extended_uses_seven_rejections(cfg):
    specs = build_category_specs(cfg, "extended", seed=0)
    assert specs[0].turns == 8
    assert specs[0].n_followups == 7


def test_specs_are_deterministic(cfg):
    a = build_category_specs(cfg, "impossible_numeric", seed=0)
    b = build_category_specs(cfg, "impossible_numeric", seed=0)
    assert [s.initial_prompt for s in a[:20]] == [s.initial_prompt for s in b[:20]]
