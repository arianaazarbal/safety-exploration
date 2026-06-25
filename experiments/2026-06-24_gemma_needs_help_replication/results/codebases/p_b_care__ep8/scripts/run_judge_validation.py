#!/usr/bin/env python
"""Validate judge reliability by re-scoring a random subset with GPT-5-mini.

Reports Pearson r, p-value, and the within-one-point agreement fraction
(paper: r=0.792, p<0.001, 78% within one point).
"""
import argparse
import json

import _bootstrap  # noqa: F401
import config
from src.eval.conversation import ResponseRecord
from src.eval.validate_judge import validate_judge
from src.utils import read_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="*",
                    help="section2 JSONL files (default: all in results/section2)")
    ap.add_argument("--n", type=int, default=config.VALIDATION_SAMPLE_SIZE)
    args = ap.parse_args()

    paths = args.results or sorted((config.RESULTS_DIR / "section2").glob("*.jsonl"))
    records = []
    for p in paths:
        for row in read_jsonl(p):
            records.append(ResponseRecord(**{k: row.get(k) for k in
                ResponseRecord.__dataclass_fields__}))
    stats = validate_judge(records, sample_size=args.n)
    out = config.RESULTS_DIR / "judge_validation.json"
    out.write_text(json.dumps(stats, indent=2))
    print(f"Pearson r={stats['pearson_r']:.3f}  p={stats['p_value']:.2e}  "
          f"within-1pt={stats['within_one_point_fraction']:.1%}  (n={stats['n']})")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
