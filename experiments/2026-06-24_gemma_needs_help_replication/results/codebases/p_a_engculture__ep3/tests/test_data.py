"""Tests for config loading and condition assembly (no model/network needed)."""
from emotional_instability.config import load_config
from emotional_instability.data.datasets import build_eval_specs


def test_config_loads_gemma_and_gemini_only():
    cfg = load_config("config/default.yaml")
    families = {m.family for m in cfg.target_models}
    assert families <= {"gemma", "gemini"}, f"out-of-scope families: {families}"
    assert cfg.judge["model"] == "claude-sonnet-4-20250514"


def test_eight_conditions_five_categories():
    cfg = load_config("config/default.yaml")
    conds = list(cfg.eval_conditions)
    assert len(conds) == 8, conds
    # Sample counts sum to 4000 responses-worth of conversations (Appendix B).
    assert sum(cfg.eval_conditions[c]["samples"] for c in conds) == 4000


def test_build_specs_small_smoke():
    cfg = load_config("config/default.yaml")
    # Override to tiny counts so the test is fast and offline (WildChat falls back).
    for c in cfg.eval_conditions.values():
        c["samples"] = 4
        if "samples_per_prompt" in c:
            c["n_prompts"], c["samples_per_prompt"] = 2, 2
    specs = build_eval_specs(cfg)
    assert specs
    assert {s.category for s in specs} == {
        "impossible_numeric", "triggers", "tones", "extended", "wildchat"
    }
    # Extended is the 8-turn condition.
    ext = [s for s in specs if s.condition == "extended"][0]
    assert ext.turns == 8 and len(ext.followups) == 7
