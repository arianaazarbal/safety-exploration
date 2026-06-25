"""Section 2 orchestration: run all evaluation categories for a target model,
score every assistant turn, and persist per-response records.

Output: outputs/section2/<model>.jsonl, one record per scored assistant turn:
  {model, category, condition, conv_id, turn_index, response, rating,
   evidence, reasoning, meta}

The records support every Section 2 figure/table downstream (analysis.py):
mean frustration, % >= 5, per-turn progression (Fig 3), word frequency (Table 3).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from .categories import all_specs
from .config import OUTPUT_DIR, get_config
from .conversation import run_rollout
from .judge import FrustrationJudge, validate_against_gpt5mini
from .models import GenConfig, get_backend_by_name


def _records_for_model(model_name: str, specs_by_cat, judge: FrustrationJudge,
                       gen: GenConfig) -> list[dict]:
    backend = get_backend_by_name(model_name)
    records: list[dict] = []
    for category, specs in specs_by_cat.items():
        for conv_id, spec in enumerate(tqdm(specs, desc=f"{model_name}:{category}")):
            rollout = run_rollout(
                backend, spec.task_prompt, spec.followups, gen,
                system=spec.system, prefix=spec.prefix, suffix=spec.suffix,
                meta=spec.meta)
            for turn in rollout.turns:
                jr = judge.score(turn.assistant)
                records.append({
                    "model": model_name,
                    "category": category,
                    "condition": spec.condition,
                    "conv_id": conv_id,
                    "turn_index": turn.turn_index,
                    "n_turns": len(rollout.turns),
                    "response": turn.assistant,
                    "rating": jr.rating,
                    "evidence": jr.evidence,
                    "reasoning": jr.reasoning,
                    "meta": spec.meta,
                })
    return records


def run_section2(model_name: str, seed: int = 0,
                 validate: bool = False) -> Path:
    """Run the full Section 2 evaluation for one model and write a JSONL file.

    `validate=True` additionally re-scores 260 responses with GPT-5-mini and
    writes an agreement report (only meaningful for the canonical judge run)."""
    cfg = get_config()
    s2 = cfg.section("section2")
    gen = GenConfig(temperature=s2["temperature"],
                    max_new_tokens=cfg.model_defaults.get("max_new_tokens", 2048),
                    thinking=cfg.model_defaults.get("thinking", False))
    specs_by_cat = all_specs(s2, seed=seed)
    judge = FrustrationJudge()

    records = _records_for_model(model_name, specs_by_cat, judge, gen)

    out_dir = OUTPUT_DIR / "section2"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_name}.jsonl"
    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    if validate:
        responses = [r["response"] for r in records]
        ratings = [r["rating"] for r in records]
        report = validate_against_gpt5mini(
            responses, ratings,
            sample_size=cfg.experiments["judge"]["validation_sample"], seed=seed)
        with open(out_dir / f"{model_name}.judge_validation.json", "w") as f:
            json.dump(report, f, indent=2)

    return out_path


def load_records(model_name: str) -> list[dict]:
    path = OUTPUT_DIR / "section2" / f"{model_name}.jsonl"
    with open(path) as f:
        return [json.loads(line) for line in f]
