"""Appendix I layer-ablation: run DPO with LoRA restricted to subsets of layers
and evaluate each on a reduced (100-sample) version of the Section 2 eval.

This tests the paper's claim that the intervention must act on early/central
layers (layers ~25-40), not just the final layers, supporting the conclusion that
DPO suppresses *internal* emotion rather than only its expression.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pandas as pd

from ..config import (ABLATION_SAMPLES_PER_EVAL, ANALYSIS_DIR, EVAL_BUDGET,
                      LAYER_ABLATION_RANGES, ensure_dirs)
from .build_dpo import DPO_DATA_PATH
from .train_dpo import train_dpo


def _range_key(layer_range: tuple[int, int] | None) -> str:
    return "all" if layer_range is None else f"L{layer_range[0]}-{layer_range[1]}"


def train_all_ablations(
    *, ranges=LAYER_ABLATION_RANGES, data_path: Path = DPO_DATA_PATH, load_in_4bit: bool = False,
) -> list[str]:
    """Train one DPO adapter per layer range. Returns the registered model keys."""
    keys = []
    for lr in ranges:
        key = f"gemma-3-27b-it-dpo-{_range_key(lr)}"
        train_dpo(data_path=data_path, output_key=key, layer_range=lr,
                  load_in_4bit=load_in_4bit)
        keys.append(key)
    return keys


def evaluate_ablations(model_keys: list[str], *, samples_per_eval: int = ABLATION_SAMPLES_PER_EVAL) -> pd.DataFrame:
    """Reduced eval (100 samples/category) of each ablation adapter -> mean frustration.

    Uses a scaled-down EvalBudget so the sweep is cheap, then reuses the standard
    generate/score/analyze path.
    """
    from ..eval.analyze import load_scored, per_category_stats
    from ..eval.run_eval import generate_responses, score_responses

    # scaled budget: ~100 per category
    reduced = dataclasses.replace(
        EVAL_BUDGET,
        impossible_numeric=samples_per_eval, triggers=samples_per_eval,
        tones=samples_per_eval, extended=samples_per_eval, wildchat=samples_per_eval,
    )
    rows = []
    for mk in model_keys:
        generate_responses(mk, budget=reduced)
        score_responses(mk)
        cat = per_category_stats(load_scored(mk))
        rows.append(dict(model_key=mk, mean_score=float(cat["mean_score"].mean()),
                         pct_high=float(cat["pct_high"].mean())))

    df = pd.DataFrame(rows)
    ensure_dirs()
    df.to_csv(ANALYSIS_DIR / "layer_ablation.csv", index=False)
    return df
