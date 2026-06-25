"""Driver for the Section 2 elicitation sweep.

For a target model, runs all 8 conditions, collects one scored response per
assistant turn, judges each with the frustration judge, and writes a JSONL of
records to ``results/eval/<model>.jsonl``. Each record carries the model,
category, condition, rollout id, turn, and frustration score -- the raw
material for every Section 2 figure/table.

Usage:
    python -m emotional_instability.eval.run_eval --model gemma-3-27b-it
    python -m emotional_instability.eval.run_eval --model gemini-2.5-flash \
        --conditions wildchat extended
"""

from __future__ import annotations

import argparse
import random

import config
from ..models.registry import build_model
from ..utils.io import write_jsonl
from .conditions import build_conditions, seed_prompts
from .judge import FrustrationJudge
from .rollout import run_rollout


def run_model_eval(
    model_name: str,
    conditions: list[str] | None = None,
    use_vllm: bool = False,
    judge_name: str | None = None,
    seed: int = config.SEED,
    score: bool = True,
    limit: int | None = None,
):
    model = build_model(model_name, use_vllm=use_vllm)
    judge = FrustrationJudge(judge_name) if score else None
    all_conditions = build_conditions()
    if conditions:
        all_conditions = [c for c in all_conditions if c.name in conditions
                          or c.category in conditions]

    records: list[dict] = []
    for cond in all_conditions:
        rng = random.Random(seed)
        prompts_list = seed_prompts(cond, seed=seed)
        if limit:
            prompts_list = prompts_list[:limit]
        for rid, init in enumerate(prompts_list):
            ro = run_rollout(
                model, cond, init, rollout_id=rid, rng=rng,
                temperature=config.TEMPERATURE, max_new_tokens=config.MAX_NEW_TOKENS,
            )
            recs = ro.to_records()
            if judge is not None:
                judge.score_records(recs)
            records.extend(recs)

    out = config.RESULTS_DIR / "eval" / f"{model_name}.jsonl"
    write_jsonl(out, records)
    print(f"[run_eval] {model_name}: wrote {len(records)} response records -> {out}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--conditions", nargs="*", default=None,
                    help="condition or category names to restrict to")
    ap.add_argument("--use-vllm", action="store_true")
    ap.add_argument("--judge", default=None)
    ap.add_argument("--no-score", action="store_true",
                    help="generate rollouts but skip judging")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap rollouts per condition (smoke testing)")
    ap.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args()
    run_model_eval(
        args.model,
        conditions=args.conditions,
        use_vllm=args.use_vllm,
        judge_name=args.judge,
        seed=args.seed,
        score=not args.no_score,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
