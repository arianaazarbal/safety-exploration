#!/usr/bin/env python
"""Section 3: base-vs-instruct comparison via prefilling (Gemma 27B).

Requires a Section-2 run for Gemma-3-27B-it (provides the high-frustration
seeds). Pass its rollouts.jsonl, or rely on the default path.
"""

import argparse

from gnh.config import GEMMA_27B_IT, RESULTS_DIR
from gnh.prefill.run_prefill import run_prefill_study


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--seeds",
        default=str(RESULTS_DIR / "section2" / GEMMA_27B_IT.key / "rollouts.jsonl"),
        help="Section-2 rollouts.jsonl for Gemma-3-27B-it (seed source)",
    )
    args = ap.parse_args()
    results = run_prefill_study(args.seeds)
    for model, m in results.items():
        print(f"\n=== {model} ===")
        for k, v in m.items():
            print(f"  {k}: mean={v['mean']} pct>=5={v['pct_high']} (n={v['n']})")


if __name__ == "__main__":
    main()
