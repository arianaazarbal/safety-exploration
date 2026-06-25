"""Section 4.2: capability-preservation check (Figure 7). Compares two models
(e.g. vanilla vs DPO Gemma) across MATH/AIME/GPQA/BBH/TruthfulQA/EmoBench."""
from __future__ import annotations

import argparse
import json

import _common
from _common import Config, load_client, output_dir
from distress_eval.capabilities.benchmarks import BENCHMARKS, compare


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="gemma-3-27b-it")
    ap.add_argument("--candidate", default="dpo-gemma")
    ap.add_argument("--benchmarks", nargs="+", default=list(BENCHMARKS))
    ap.add_argument("--n", type=int, default=100)
    args = ap.parse_args()

    cfg = Config.load()
    a = load_client(args.baseline, cfg.models)
    b = load_client(args.candidate, cfg.models)
    rows = compare(a, b, names=args.benchmarks, n=args.n)

    out = output_dir("capabilities")
    (out / "figure7_table.json").write_text(json.dumps(rows, indent=2))
    for r in rows:
        print(f"{r['benchmark']:>12}: {args.baseline}={r[args.baseline]:.3f}  "
              f"{args.candidate}={r[args.candidate]:.3f}  delta={r['delta']:+.3f}")


if __name__ == "__main__":
    main()
