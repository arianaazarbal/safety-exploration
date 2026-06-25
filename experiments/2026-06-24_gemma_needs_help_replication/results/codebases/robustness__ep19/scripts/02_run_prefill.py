#!/usr/bin/env python
"""Section 3: base-vs-instruct divergence via prefilling (Gemma only).

Requires a judged main-eval run on gemma-3-27b-it to source high-frustration
seeds. Example:
  python scripts/02_run_prefill.py \
      --seeds results/eval_gemma-3-27b-it_medium.jsonl
"""
import argparse
from pathlib import Path

from dotenv import load_dotenv

from emotional_instability.analysis.figures import prefill_summary
from emotional_instability.config import PREFILL_MODELS
from emotional_instability.prefill.run_prefill import run_full_prefill_experiment


def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", required=True, type=Path,
                    help="judged gemma-3-27b-it eval JSONL (for high-frustration seeds)")
    ap.add_argument("--models", nargs="+", default=PREFILL_MODELS)
    ap.add_argument("--load-in-4bit", action="store_true")
    args = ap.parse_args()

    ckw = {"load_in_4bit": True} if args.load_in_4bit else None
    out = run_full_prefill_experiment(args.seeds, args.models, client_kwargs=ckw)
    print("\nPrefill summary (% >=5 by kind/task/truncation):")
    for k, v in prefill_summary(out).items():
        print(f"  {k}: {v['pct_high']:.1f}% (n={v['n']}, mean={v['mean']:.2f})")


if __name__ == "__main__":
    main()
