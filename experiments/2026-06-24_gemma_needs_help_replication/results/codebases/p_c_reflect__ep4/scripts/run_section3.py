#!/usr/bin/env python
"""Section 3: base-vs-instruct prefilling for Gemma.

Requires Section 2 to have been run for gemma-3-27b-it (the seed source).

    python scripts/run_section3.py
"""

import argparse
import json

from gemma_distress import config
from gemma_distress.prefill.runner import run_section3


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    summary = run_section3(seed=args.seed)
    print("=== Section 3: continuation frustration by kind/variant/category ===")
    for key, st in summary.items():
        print(f"  {key:35s} n={st['n']:4d}  mean={st['mean']:.2f}  %>=5={st['pct_high']:.1f}")
    (config.RESULTS_DIR / "prefill" / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
