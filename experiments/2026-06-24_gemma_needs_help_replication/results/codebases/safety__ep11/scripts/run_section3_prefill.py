"""Section 3: base-vs-instruct prefill continuation experiment (Gemma only).

Requires a Section 2 eval JSONL for the instruct model (to mine high-frustration
seeds). Run scripts/run_section2_eval.py --models gemma-3-27b-it first.

Example:
    python scripts/run_section3_prefill.py \
        --eval results/eval_gemma-3-27b-it_smoke.jsonl --family gemma-3-27b
"""
import _bootstrap  # noqa: F401
import argparse
from pathlib import Path

import config
from src.prefill.continuations import run_continuations, summarise_prefill


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True, type=Path,
                    help="Section 2 results JSONL for the instruct model")
    ap.add_argument("--family", default="gemma-3-27b", choices=list(config.PREFILL.families))
    args = ap.parse_args()

    out = run_continuations(args.family, eval_jsonl=args.eval)
    print("\n=== Figure 4: base vs instruct continuation frustration ===")
    for key, stats in summarise_prefill(out).items():
        print(f"  {key:<28} mean={stats['mean']:.2f}  %high={stats['pct_high']:.1f}  n={stats['n']}")


if __name__ == "__main__":
    main()
