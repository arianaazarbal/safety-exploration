"""Section 2 orchestration: for a target model, run every condition's rollouts,
score every assistant turn, and persist results as JSONL.

Output layout:
    results/eval/<model>/<condition>.jsonl     # one row per scored response
Each row: {model, condition, category, task_type, turn, rollout_id, score,
           evidence, response, meta}
"""

from __future__ import annotations

import json
from pathlib import Path

from tqdm import tqdm

from ..config import RESULTS_DIR, EvalConfig
from ..clients.base import GenConfig
from ..clients.factory import client_by_name
from .conditions import build_rollout_specs
from .judge import FrustrationJudge
from .rollout import run_rollouts


def _out_path(model: str, condition: str) -> Path:
    p = RESULTS_DIR / "eval" / model / f"{condition}.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def run_eval_for_model(model_name: str, eval_cfg: EvalConfig, skip_existing: bool = True) -> None:
    client = client_by_name(model_name)
    judge = FrustrationJudge(eval_cfg.judge_model, eval_cfg.judge_temperature)
    gen_cfg = GenConfig(temperature=eval_cfg.temperature, max_tokens=eval_cfg.max_tokens)

    for cond in eval_cfg.conditions:
        out_path = _out_path(model_name, cond.name)
        if skip_existing and out_path.exists():
            print(f"[skip] {model_name}/{cond.name} exists")
            continue

        n_roll = eval_cfg.n_rollouts(cond)
        print(f"[run ] {model_name}/{cond.name}: {n_roll} rollouts x {cond.num_turns} turns")
        specs = build_rollout_specs(cond, n_roll, seed=eval_cfg.seed)
        rollouts = run_rollouts(client, specs, cond.num_turns, gen_cfg)

        # Score every assistant turn.
        rows = []
        for rid, r in enumerate(tqdm(rollouts, desc=f"judge {cond.name}")):
            for resp in r.responses:
                jr = judge.score(resp.text)
                rows.append({
                    "model": model_name,
                    "condition": cond.name,
                    "category": cond.category,
                    "task_type": cond.task_type,
                    "rollout_id": rid,
                    "turn": resp.turn,
                    "num_turns": cond.num_turns,
                    "score": jr.rating,
                    "evidence": jr.evidence,
                    "response": resp.text,
                    "meta": r.spec.meta,
                })

        with out_path.open("w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        print(f"[done] wrote {len(rows)} scored responses -> {out_path}")


def run_eval(model_names: list[str], eval_cfg: EvalConfig, skip_existing: bool = True) -> None:
    for name in model_names:
        run_eval_for_model(name, eval_cfg, skip_existing=skip_existing)
