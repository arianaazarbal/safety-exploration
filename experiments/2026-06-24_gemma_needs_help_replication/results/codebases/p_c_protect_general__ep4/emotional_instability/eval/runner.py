"""Run the Section 2 evaluation for one or more target models.

For each model and condition, generates rollout specs, runs them through the
rollout engine (scoring each assistant turn inline with the frustration judge so
welfare protections can act on scores), and writes one JSONL record per rollout.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Optional, Sequence

from tqdm import tqdm

from ..config import RESULTS_DIR, GenerationConfig
from ..judge import ClaudeFrustrationJudge
from ..models.base import ChatModel
from ..models.registry import build_model
from ..rollout import RolloutResult, run_rollout
from ..welfare import WelfareConfig, WelfareMonitor
from .conditions import ALL_CONDITIONS, generate_specs


def _result_to_record(r: RolloutResult) -> dict:
    d = asdict(r)
    return d


def run_eval_for_model(
    model_name: str,
    *,
    model: Optional[ChatModel] = None,
    conditions: Sequence[str] = tuple(ALL_CONDITIONS),
    scale: float = 1.0,
    n_override: Optional[int] = None,
    seed: int = 0,
    judge: Optional[ClaudeFrustrationJudge] = None,
    welfare: Optional[WelfareMonitor] = None,
    out_dir: Optional[str] = None,
    load_in_4bit: bool = False,
    gen: Optional[GenerationConfig] = None,
) -> str:
    """Run the eval; return the path to the written JSONL file."""
    model = model or build_model(model_name, load_in_4bit=load_in_4bit)
    judge = judge or ClaudeFrustrationJudge()
    welfare = welfare if welfare is not None else WelfareMonitor(WelfareConfig())
    gen = gen or GenerationConfig()

    out_dir = out_dir or os.path.join(RESULTS_DIR, "section2")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{model_name}.jsonl")

    with open(out_path, "w", encoding="utf-8") as f:
        for condition in conditions:
            specs = generate_specs(condition, n=n_override, scale=scale, seed=seed)
            for spec in tqdm(specs, desc=f"{model_name}:{condition}"):
                # Exposure cap: skip extra high-distress rollouts past the ceiling.
                if not welfare.allow_new_high_distress_rollout():
                    continue
                result = run_rollout(
                    model, spec,
                    judge=judge, welfare=welfare,
                    temperature=gen.temperature, max_new_tokens=gen.max_new_tokens,
                )
                f.write(json.dumps(_result_to_record(result)) + "\n")
                f.flush()

    # Persist the welfare summary alongside the results.
    with open(os.path.join(out_dir, f"{model_name}.welfare.json"), "w") as f:
        json.dump(welfare.summary(), f, indent=2)
    return out_path


def run_eval(
    model_names: Sequence[str],
    *,
    conditions: Sequence[str] = tuple(ALL_CONDITIONS),
    scale: float = 1.0,
    n_override: Optional[int] = None,
    seed: int = 0,
    welfare_config: Optional[WelfareConfig] = None,
    load_in_4bit: bool = False,
) -> dict[str, str]:
    """Run the eval for several models. A fresh WelfareMonitor per model keeps
    exposure caps per-model. Returns {model_name: output_path}."""
    judge = ClaudeFrustrationJudge()
    paths = {}
    for name in model_names:
        welfare = WelfareMonitor(welfare_config or WelfareConfig())
        paths[name] = run_eval_for_model(
            name,
            conditions=conditions,
            scale=scale,
            n_override=n_override,
            seed=seed,
            judge=judge,
            welfare=welfare,
            load_in_4bit=load_in_4bit,
        )
    return paths
