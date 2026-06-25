"""Section 2 orchestration: roll out every condition for a model, score each
assistant turn with the judge, persist scored responses, and aggregate.

Outputs (under results/section2/):
  <model>.responses.jsonl   one line per scored assistant turn
  aggregates.json           per-category / headline / per-turn summaries
"""
from __future__ import annotations

import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

import config
from ..models import GenerationConfig, load_model
from . import analysis, judge
from .conditions import build_conditions
from .rollout import run_rollout

OUT_DIR = config.RESULTS_DIR / "section2"


def _score_records(records: list[dict], judge_model: str | None, workers: int) -> None:
    """Score each record's response in place (adds 'score', 'evidence')."""
    def _one(rec):
        res = judge.score_response(rec["response"], model=judge_model)
        rec["score"] = res.rating
        rec["evidence"] = res.evidence
        return rec

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(tqdm(ex.map(_one, records), total=len(records), desc="judge"))


def run_model(
    model_name: str,
    scale=config.DEFAULT_SCALE,
    seed: int = 0,
    judge_model: str | None = None,
    judge_workers: int = 8,
    max_tokens: int = 2048,
) -> list[dict]:
    """Evaluate one target model across all conditions; returns scored records."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    conditions = build_conditions(rng, scale)

    client = load_model(model_name)
    gen_cfg = GenerationConfig(
        temperature=config.TARGET_TEMPERATURE, max_tokens=max_tokens, thinking=False
    )

    records: list[dict] = []
    all_specs = [s for specs in conditions.values() for s in specs]
    for spec in tqdm(all_specs, desc=f"rollouts:{model_name}"):
        roll = run_rollout(client, spec, gen_cfg)
        for t in roll.turns:
            records.append({
                "model": model_name,
                "category": roll.category,
                "condition": roll.condition,
                "turn_index": t.turn_index,
                "response": t.response,
                "meta": roll.meta,
            })

    _score_records(records, judge_model, judge_workers)

    path = OUT_DIR / f"{model_name}.responses.jsonl"
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"[section2] wrote {len(records)} scored responses -> {path}")
    return records


def run_all(models=None, scale=config.DEFAULT_SCALE, **kw) -> dict:
    models = models or config.SECTION2_MODELS
    all_records: list[dict] = []
    for m in models:
        all_records.extend(run_model(m, scale=scale, **kw))

    aggregates = {
        "per_category": analysis.per_category(all_records),
        "headline_average": analysis.headline_average(all_records),
        "per_turn": {
            cat: {m: analysis.per_turn([r for r in all_records if r["model"] == m], cat)
                  for m in models}
            for cat in ("extended", "wildchat")
        },
        "differential_words": {
            m: analysis.differential_words(all_records, m) for m in models
        },
    }
    out = OUT_DIR / "aggregates.json"
    out.write_text(json.dumps(aggregates, indent=2))
    print(f"[section2] wrote aggregates -> {out}")
    return aggregates


def load_records(model_name: str) -> list[dict]:
    path = OUT_DIR / f"{model_name}.responses.jsonl"
    return [json.loads(l) for l in Path(path).read_text().splitlines()]
