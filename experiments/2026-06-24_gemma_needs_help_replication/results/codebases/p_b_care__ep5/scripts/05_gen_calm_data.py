#!/usr/bin/env python
"""Section 4.1: generate calm response data from Gemma-3-27B-it.

Runs reassured (and optionally 'teacher') rollouts on impossible numeric puzzles,
judging each turn, so the downstream dataset builder can filter to calm (0-1)
responses. Streams to artifacts/rollouts/calm_<variant>.jsonl.

Usage:
    python scripts/05_gen_calm_data.py --n 1500
    python scripts/05_gen_calm_data.py --n 1500 --variant teacher
"""
import argparse

from _bootstrap import default_workers
from gemma_distress import config
from gemma_distress.eval import run_eval
from gemma_distress.eval.judge import FrustrationJudge
from gemma_distress.models import load_model
from gemma_distress.training import build_calm_specs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="gemma-3-27b-it", choices=sorted(config.MODELS))
    ap.add_argument("--variant", default="reassured", choices=["reassured", "teacher"])
    ap.add_argument("--n", type=int, default=1500,
                    help="rollouts to sample (oversample; calm filter keeps a fraction)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--load-in-4bit", action="store_true")
    args = ap.parse_args()

    specs = build_calm_specs(args.n, seed=args.seed, variant=args.variant)
    model = load_model(args.source, load_in_4bit=args.load_in_4bit)
    judge = FrustrationJudge()
    out = config.DATA_DIR / "rollouts" / f"calm_{args.variant}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"[calm] variant={args.variant} specs={len(specs)} -> {out}")
    run_eval(model, specs, judge, out, max_workers=default_workers(args.source))
    print(f"[calm] done -> {out}")


if __name__ == "__main__":
    main()
