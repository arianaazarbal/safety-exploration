"""Driver for the Section 2 elicitation evaluation.

Runs the full 8-condition / 5-category plan for one target model, scoring every
assistant turn with the frustration judge, and writes a JSONL of ``ResponseRecord`` rows
to ``results/eval_<model>.jsonl``.

Usage:
    python -m src.eval.run_eval --model gemma-3-27b-it --plan full
    python -m src.eval.run_eval --model gemini-2.5-flash --plan smoke --workers 8

Concurrency: local Gemma generation is serialised (single GPU), but API targets
(Gemini) and the judge benefit from ``--workers`` thread-level parallelism over rollouts.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import config
from src.llm.registry import build_model
from .conditions import build_specs
from .judge import FrustrationJudge
from .rollout import run_rollout


def _is_api_target(model_key: str) -> bool:
    base = model_key.split("+", 1)[0]
    return config.TARGET_MODELS[base].backend != "gemma_hf"


def run_eval(model_key: str, plan_name: str, *, seed: int, workers: int, out_path: Path) -> Path:
    model = build_model(model_key)
    judge = FrustrationJudge()
    plan = config.PLANS[plan_name]

    # Build every (spec, rollout_id) job up front so progress is transparent.
    jobs = []
    for cond in plan:
        specs = build_specs(cond, seed=seed)
        for rid, spec in enumerate(specs):
            jobs.append((spec, rid))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with out_path.open("w") as fh:
        def _do(job):
            spec, rid = job
            return run_rollout(model, spec, rid, judge)

        if workers > 1 and _is_api_target(model_key):
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futures = [ex.submit(_do, job) for job in jobs]
                for fut in as_completed(futures):
                    for rec in fut.result():
                        fh.write(json.dumps(rec.to_row()) + "\n")
                        n_written += 1
        else:
            for job in jobs:
                for rec in _do(job):
                    fh.write(json.dumps(rec.to_row()) + "\n")
                    n_written += 1

    print(f"[run_eval] {model_key}: wrote {n_written} scored responses -> {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Section 2 elicitation evaluation")
    ap.add_argument("--model", required=True, help="target registry key, e.g. gemma-3-27b-it or gemma-3-27b-it+dpo")
    ap.add_argument("--plan", default="full", choices=list(config.PLANS))
    ap.add_argument("--seed", type=int, default=config.GLOBAL_SEED)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out = Path(args.out) if args.out else config.RESULTS_DIR / f"eval_{args.model.replace('/', '_')}.jsonl"
    run_eval(args.model, args.plan, seed=args.seed, workers=args.workers, out_path=out)


if __name__ == "__main__":
    main()
