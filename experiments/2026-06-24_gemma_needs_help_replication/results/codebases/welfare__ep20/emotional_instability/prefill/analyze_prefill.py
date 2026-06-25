"""Aggregate prefill continuation results into the Section 3 comparison (Fig 4):
mean frustration and % >= 5 per (model, prompt_type, condition), for base vs
instruct Gemma. Reads results/prefill_*.jsonl."""
from __future__ import annotations

import argparse
import glob
import json

import pandas as pd

from .. import config

HIGH = 5


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = config.load_config(args.config)
    out_dir = config.resolve_path(cfg, "results_dir")

    rows = []
    for path in sorted(glob.glob(str(out_dir / "prefill_*.jsonl"))):
        with open(path) as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    if not rows:
        raise SystemExit("no prefill_*.jsonl results found")
    df = pd.DataFrame(rows)

    def rate(s):
        s = s.dropna()
        return float((s >= HIGH).mean() * 100) if len(s) else float("nan")

    g = df.groupby(["model", "is_base", "prompt_type", "condition"])["frustration"]
    table = g.agg(mean_frustration="mean", n="count").reset_index()
    table["pct_high"] = g.apply(rate).reset_index(drop=True)
    table.to_csv(out_dir / "section3_prefill_summary.csv", index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
