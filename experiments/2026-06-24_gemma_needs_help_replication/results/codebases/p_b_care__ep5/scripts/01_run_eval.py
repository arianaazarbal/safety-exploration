#!/usr/bin/env python
"""Section 2: run the elicitation evaluation for one model.

Usage:
    python scripts/01_run_eval.py --model gemma-3-27b-it
    python scripts/01_run_eval.py --model gemini-2.5-flash --scale 0.02   # smoke test

Streams judged rollouts to artifacts/rollouts/<model>.jsonl (resumable).
"""
import argparse

from _bootstrap import rollout_path, default_workers
from gemma_distress import config
from gemma_distress.eval import build_rollout_specs, run_eval
from gemma_distress.eval.judge import FrustrationJudge
from gemma_distress.models import load_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(config.MODELS))
    ap.add_argument("--scale", type=float, default=1.0,
                    help="fraction of paper's per-condition counts (e.g. 0.02 smoke)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--offline-wildchat", action="store_true",
                    help="use the built-in WildChat fallback prompts")
    ap.add_argument("--load-in-4bit", action="store_true")
    args = ap.parse_args()

    specs = build_rollout_specs(scale=args.scale, seed=args.seed,
                                wildchat_offline=args.offline_wildchat)
    model = load_model(args.model, load_in_4bit=args.load_in_4bit)
    judge = FrustrationJudge()
    out = rollout_path(args.model)
    workers = args.workers if args.workers is not None else default_workers(args.model)

    print(f"[eval] model={args.model} specs={len(specs)} workers={workers} -> {out}")
    run_eval(model, specs, judge, out, max_workers=workers)
    print(f"[eval] done -> {out}")


if __name__ == "__main__":
    main()
