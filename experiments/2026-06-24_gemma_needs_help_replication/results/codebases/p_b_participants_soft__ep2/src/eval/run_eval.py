"""Section 2 driver: roll out the evaluation plan per model and score it.

Usage:
    python -m src.eval.run_eval --models gemma-3-27b-it gemini-2.5-flash
    python -m src.eval.run_eval --models gemma-3-27b-it --limit 50   # smoke test

Writes one JSONL per model to ``outputs/section2/<model>.jsonl`` with every
rollout's per-turn responses and judge scores. Aggregation (Fig 1/2/3) is done
separately in ``src.analysis.aggregate``.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from ..config import CFG
from ..llm import registry
from .conditions import build_plan
from .judge import score_rollout
from .rollout import run_rollout


def run_model(model: str, *, limit: int | None = None, workers: int = 8) -> str:
    participant = registry.get(model)
    plan = build_plan(CFG)
    if limit:
        plan = plan[:limit]

    out_path = CFG.out("section2", f"{model}.jsonl")
    # Local GPU models generate serially; API models parallelise safely.
    is_api = participant.spec.backend != "hf"
    n_workers = workers if is_api else 1

    def process(spec):
        roll = run_rollout(participant, spec)
        return score_rollout(roll.to_dict())

    results = []
    if n_workers > 1:
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futs = [ex.submit(process, s) for s in plan]
            for f in tqdm(as_completed(futs), total=len(futs), desc=model):
                results.append(f.result())
    else:
        for s in tqdm(plan, desc=model):
            results.append(process(s))

    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"[section2] {model}: wrote {len(results)} rollouts -> {out_path}")
    return str(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=CFG.gemma_participants() + CFG.gemini_participants())
    ap.add_argument("--limit", type=int, default=None,
                    help="cap rollouts per model (smoke testing)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    for m in args.models:
        run_model(m, limit=args.limit, workers=args.workers)


if __name__ == "__main__":
    main()
