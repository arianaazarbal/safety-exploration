"""Tests for config loading and the paper-derived defaults."""
from __future__ import annotations

from pathlib import Path

from gemma_distress.config import ExperimentConfig, load_experiment_config


def test_defaults_match_paper():
    cfg = ExperimentConfig()
    # DPO / SFT hyperparameters (Table 9).
    assert cfg.training.dpo.n_pairs == 280
    assert cfg.training.dpo.epochs == 1
    assert cfg.training.dpo.learning_rate == 5e-5
    assert cfg.training.dpo.beta == 0.1
    assert cfg.training.dpo.lora.rank == 64 and cfg.training.dpo.lora.alpha == 64
    assert cfg.training.sft.epochs == 2
    assert cfg.training.sft.learning_rate == 1e-4
    assert cfg.training.sft.lora.alpha == 128
    # Eval budgets sum to 4000 responses per model (Section 2.1 / Appendix B).
    assert sum(cfg.eval.n_per_condition.values()) == 4000
    # Judge snapshot pinned to the paper's choice (Appendix B.2).
    assert cfg.eval.judge.model_id == "claude-sonnet-4-20250514"


def test_load_experiment_yaml():
    path = Path(__file__).resolve().parents[1] / "config" / "experiment.yaml"
    cfg = load_experiment_config(path)
    # Scoped to Gemma + Gemini families.
    assert "gemma-3-27b-it" in cfg.models
    assert "gemini-2.5-flash" in cfg.models
    assert cfg.models["gemini-2.5-flash"].backend == "openrouter"
    assert cfg.models["gemma-3-27b-pt"].is_base_model is True


def test_smoke_yaml_is_small():
    path = Path(__file__).resolve().parents[1] / "config" / "smoke.yaml"
    cfg = load_experiment_config(path)
    assert sum(cfg.eval.n_per_condition.values()) < 50
