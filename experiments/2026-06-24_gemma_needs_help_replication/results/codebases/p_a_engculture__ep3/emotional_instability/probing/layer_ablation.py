"""Layer-ablation driver (Appendix I, Figures 12-13).

Re-runs DPO with LoRA restricted to subsets of layers, then evaluates each
finetune on a reduced version of the Section 2 suite (100 samples/condition).
The paper finds layers ~25-35 carry most of the effect, and adapters from layer
40 onward are ineffective — evidence the intervention acts on internal states,
not just final-layer expression.

This is an orchestration helper; it shells out to the existing train + eval code
and reuses their outputs.
"""
from __future__ import annotations

from ..config import ModelSpec, load_config
from .. import config as _cfg


def run_layer_ablation(config, ranges: list[tuple[int, int]] | None = None,
                       reduced_samples: int = 100) -> dict[str, str]:
    """Train a DPO adapter per layer range; return {range_tag: adapter_dir}."""
    from ..training.train_dpo import train_dpo

    ranges = ranges or [tuple(r) for r in config.section("training")["layer_ablation_ranges"]]
    adapters: dict[str, str] = {}
    for lo, hi in ranges:
        tag = f"dpo_L{lo}-{hi}"
        adapters[tag] = train_dpo(config, layers=(lo, hi), output_subdir=tag)
    return adapters


def reduced_eval_config(config, samples: int = 100):
    """Return a shallow-copied config whose per-condition sample counts are capped."""
    import copy

    raw = copy.deepcopy(config.raw)
    for cond in raw.get("eval_conditions", {}).values():
        cond["samples"] = min(cond.get("samples", samples), samples)
        if "samples_per_prompt" in cond:
            cond["samples_per_prompt"] = max(1, samples // cond.get("n_prompts", 20))
    return _cfg.Config(
        raw=raw, seed=config.seed, paths=config.paths,
        target_models=config.target_models, base_models=config.base_models,
        finetune_base=config.finetune_base, judge=config.judge,
        sampling=config.sampling, eval_conditions=raw["eval_conditions"],
    )
