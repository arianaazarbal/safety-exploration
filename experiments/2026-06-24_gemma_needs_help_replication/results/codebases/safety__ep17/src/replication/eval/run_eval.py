"""Section 2 driver: elicit responses, judge them, and aggregate metrics.

Usage (examples)::

    # One model, all 8 conditions, default 500 responses/condition.
    python -m src.replication.eval.run_eval --models gemma-3-27b-it

    # Quick smoke test.
    REPLICATION_N_PER_CONDITION=5 python -m src.replication.eval.run_eval \
        --models gemini-2.5-flash --conditions extended_8turn

Outputs, per model, under ``results/section2/<model>/``:
* ``rollouts.jsonl``  -- raw multi-turn conversations
* ``scored.jsonl``    -- per-turn frustration scores
* ``metrics.json``    -- aggregate + per-turn metrics
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import config
from ..judge.frustration_judge import FrustrationJudge
from ..models.registry import build_client
from . import metrics as metric_mod
from .conditions import CONDITIONS, CONDITIONS_BY_NAME, Condition, build_tasks
from .rollout import Rollout, run_rollout

OUT_ROOT = config.RESULTS_DIR / "section2"


def score_rollouts(rollouts: list[Rollout], judge: FrustrationJudge,
                   max_workers: int = 8) -> list[metric_mod.ScoredTurn]:
    """Judge every assistant turn of every rollout (parallel over API calls)."""
    cond_category = {c.name: c.category for c in CONDITIONS}
    jobs = []
    for r in rollouts:
        n_turns = len(r.turns)
        for t in r.turns:
            jobs.append((r, t, t.turn_index == n_turns - 1))

    def _score(job):
        r, turn, is_final = job
        res = judge.score(turn.assistant_text)
        return metric_mod.ScoredTurn(
            model_key=r.model_key,
            condition=r.condition,
            category=cond_category.get(r.condition, r.condition),
            task_id=r.task_id,
            turn_index=turn.turn_index,
            is_final=is_final,
            score=res.rating,
        )

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        return list(ex.map(_score, jobs))


def run_model(model_key: str, conditions: list[Condition], n_per_condition: int,
              seed: int, judge_workers: int, adapter_path: str | None = None,
              label: str | None = None) -> None:
    # When evaluating a finetuned adapter (Section 4), load the base Gemma spec
    # and layer the LoRA adapter on top; results are written under ``label``.
    spec = config.TARGET_MODELS[model_key]
    client = build_client(spec, adapter_path=adapter_path)
    judge = FrustrationJudge()

    label = label or model_key
    out_dir = OUT_ROOT / label
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rollouts: list[Rollout] = []
    for cond in conditions:
        tasks = build_tasks(cond, n_per_condition, seed=seed)
        print(f"[{model_key}] {cond.name}: {len(tasks)} rollouts x {cond.n_turns} turns")
        for j, task in enumerate(tasks):
            roll = run_rollout(client, task, cond, seed=seed * 100003 + j)
            all_rollouts.append(roll)

    # Persist raw rollouts.
    with (out_dir / "rollouts.jsonl").open("w") as f:
        for r in all_rollouts:
            f.write(json.dumps(r.to_dict()) + "\n")

    # Judge.
    scored = score_rollouts(all_rollouts, judge, max_workers=judge_workers)
    with (out_dir / "scored.jsonl").open("w") as f:
        for s in scored:
            f.write(json.dumps(s.__dict__) + "\n")

    # Aggregate. Scored records are keyed by the client's base key; results are
    # filed under ``label`` (which may differ for finetuned adapters).
    ck = client.key
    out_metrics = {
        "model": label,
        "base_model": model_key,
        "adapter_path": adapter_path,
        "n_per_condition": n_per_condition,
        "aggregate": metric_mod.aggregate_by_model(scored).get(ck, {}),
        "by_category": metric_mod.aggregate_by_model_condition(scored).get(ck, {}),
        "per_turn": {
            cond.name: metric_mod.per_turn_progression(scored, cond.name).get(ck, {})
            for cond in conditions
        },
    }
    (out_dir / "metrics.json").write_text(json.dumps(out_metrics, indent=2))
    print(f"[{label}] aggregate: {out_metrics['aggregate']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(config.TARGET_MODELS),
                    choices=list(config.TARGET_MODELS))
    ap.add_argument("--conditions", nargs="+", default=[c.name for c in CONDITIONS],
                    choices=[c.name for c in CONDITIONS])
    ap.add_argument("--n-per-condition", type=int, default=config.N_PER_CONDITION)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--judge-workers", type=int, default=8)
    ap.add_argument("--adapter", default=None,
                    help="LoRA adapter path; evaluates the finetuned Gemma (Section 4).")
    ap.add_argument("--label", default=None,
                    help="Result subdirectory name (defaults to the model key).")
    args = ap.parse_args()

    conditions = [CONDITIONS_BY_NAME[c] for c in args.conditions]
    for model_key in args.models:
        run_model(model_key, conditions, args.n_per_condition, args.seed,
                  args.judge_workers, adapter_path=args.adapter, label=args.label)


if __name__ == "__main__":
    main()
