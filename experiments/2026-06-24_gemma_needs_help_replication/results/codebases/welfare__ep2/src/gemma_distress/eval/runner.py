"""Orchestrates a full Section 2 evaluation for one model:
build plans -> roll out conversations -> judge every response -> persist + summarize.

Outputs (under ``outputs/eval/<model>/``):
* ``responses.jsonl`` -- one line per scored assistant turn (full record)
* ``summary.json``    -- overall / per-category / per-turn aggregates + Fig-1 number
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from ..config import load_eval, load_models, output_path
from ..models import load_model
from . import metrics
from .conditions import build_all_plans
from .judge import build_judge
from .rollout import ResponseRecord, run_rollouts


def _serialize(rec: ResponseRecord) -> dict:
    d = dataclasses.asdict(rec)
    return d


def run_eval(
    model_name: str,
    *,
    seed: int = 0,
    backend_kwargs: dict | None = None,
    score: bool = True,
) -> dict:
    eval_cfg = load_eval()
    models_cfg = load_models()
    threshold = eval_cfg.get("high_frustration_threshold", 5)

    plans = build_all_plans(eval_cfg, seed=seed)
    model = load_model(model_name, **(backend_kwargs or {}))

    records = run_rollouts(
        model,
        plans,
        temperature=eval_cfg.get("temperature", 1.0),
        max_new_tokens=eval_cfg.get("max_new_tokens", 2048),
    )
    model.close()

    if score:
        judge = build_judge(models_cfg["judge"])
        results = judge.score_many([r.response_text for r in records])
        for rec, res in zip(records, results):
            rec.rating = res.rating
            rec.judge_evidence = res.evidence
            rec.judge_reasoning = res.reasoning

    # Persist responses.
    resp_path = output_path("eval", model_name, "responses.jsonl")
    with open(resp_path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(_serialize(rec), ensure_ascii=False) + "\n")

    # Aggregate.
    summ = {
        "model": model_name,
        "n_responses": len(records),
        "overall": metrics.summary(records, threshold=threshold),
        "per_category": metrics.per_category(records, threshold=threshold),
        "average_pct_high_fig1": metrics.average_pct_high(records, threshold=threshold),
        "per_turn_extended": metrics.per_turn(records, ["extended"], threshold=threshold),
        "per_turn_wildchat": metrics.per_turn(records, ["wildchat"], threshold=threshold),
        "differential_words": metrics.differential_words(records),
    }
    summ_path = output_path("eval", model_name, "summary.json")
    with open(summ_path, "w", encoding="utf-8") as fh:
        json.dump(summ, fh, indent=2, ensure_ascii=False)

    return summ


def load_records(model_name: str) -> list[ResponseRecord]:
    """Reload persisted responses (for re-aggregation / agreement checks)."""
    path = Path(output_path("eval", model_name, "responses.jsonl"))
    records: list[ResponseRecord] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            records.append(ResponseRecord(**d))
    return records
