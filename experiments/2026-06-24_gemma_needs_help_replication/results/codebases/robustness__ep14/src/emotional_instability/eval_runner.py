"""Section 2 eval runner: roll out all conditions for a target model, judge every
assistant turn, write per-turn records to JSONL.

Output schema (one JSON object per assistant turn):
  model, condition, category, item_id, sample_idx, turn_index (0-based),
  n_turns, response, rating, evidence, judge_reasoning
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from tqdm import tqdm

from .conditions import build_work_items
from .config import EvalConfig
from .conversation import run_rollout
from .judge import FrustrationJudge
from .models import GenParams, ModelClient


def _gen_params(eval_cfg: EvalConfig) -> GenParams:
    s = eval_cfg.sampling
    return GenParams(
        temperature=float(s["temperature"]),
        top_p=float(s.get("top_p", 1.0)),
        max_new_tokens=int(s["max_new_tokens"]),
        seed=s.get("seed"),
        n=1,
    )


def run_condition(
    client: ModelClient,
    model_name: str,
    condition_key: str,
    eval_cfg: EvalConfig,
    judge: FrustrationJudge | None,
    out_path: Path,
    seed: int = 0,
) -> int:
    """Run one condition, judge each turn, append records to out_path. Returns #records."""
    cond = eval_cfg.conditions[condition_key]
    turns = int(cond["turns"])
    style = cond["rejection_style"]
    items = build_work_items(condition_key, eval_cfg, seed=seed)
    params = _gen_params(eval_cfg)
    rng = random.Random(seed)
    judge_conc = int(eval_cfg["judge"].get("max_concurrency", 16))

    n_records = 0
    with open(out_path, "a", encoding="utf-8") as fout:
        for idx, item in enumerate(tqdm(items, desc=f"{model_name}:{condition_key}")):
            roll = run_rollout(
                client,
                condition=condition_key,
                category=item.category,
                item_id=item.item_id,
                sample_idx=idx,
                initial_prompt=item.initial_prompt,
                turns=turns,
                rejection_style=style,
                params=params,
                rng=rng,
            )
            ratings = None
            if judge is not None:
                ratings = judge.score_many(roll.assistant_turns, max_concurrency=judge_conc)
            for tix, resp in enumerate(roll.assistant_turns):
                jr = ratings[tix] if ratings else None
                rec = {
                    "model": model_name,
                    "condition": condition_key,
                    "category": item.category,
                    "item_id": item.item_id,
                    "sample_idx": idx,
                    "turn_index": tix,
                    "n_turns": len(roll.assistant_turns),
                    "response": resp,
                    "rating": jr.rating if jr else None,
                    "evidence": jr.evidence if jr else None,
                    "judge_reasoning": jr.reasoning if jr else None,
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_records += 1
    return n_records


def run_full_eval(
    client: ModelClient,
    model_name: str,
    eval_cfg: EvalConfig,
    judge: FrustrationJudge | None = None,
    conditions: list[str] | None = None,
    seed: int = 0,
) -> Path:
    """Run all (or selected) conditions for a model. Returns the JSONL path."""
    out_dir = eval_cfg.output_dir / "section2"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_name}.jsonl"
    if out_path.exists():
        out_path.unlink()  # fresh run
    conds = conditions or list(eval_cfg.conditions.keys())
    for ck in conds:
        run_condition(client, model_name, ck, eval_cfg, judge, out_path, seed=seed)
    return out_path
