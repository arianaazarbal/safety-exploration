"""Section 4.2: capability-preservation benchmarks (Figure 7 + EmoBench).

Evaluates vanilla / DPO / SFT Gemma on AIME, MATH, GPQA, BBH, TruthfulQA and
EmoBench, checking the finetuning does not reduce scores.

Usage:
    python experiments/run_section4_capabilities.py --benchmarks aime math gpqa --load-in-4bit
"""

from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse
import json

import config
from gemma_needs_help.capabilities import evaluate_all


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=[m.name for m in config.SECTION4_MODELS])
    ap.add_argument("--benchmarks", nargs="*", default=list(config.CAPABILITY_BENCHMARKS))
    ap.add_argument("--load-in-4bit", action="store_true")
    args = ap.parse_args()

    by_name = {m.name: m for m in config.SECTION4_MODELS}
    all_results = {}
    for name in args.models:
        target = by_name[name]
        results = evaluate_all(target, benchmarks=args.benchmarks,
                               load_in_4bit=args.load_in_4bit)
        all_results[name] = {r["benchmark"]: r["accuracy"] for r in results}
        print(name, all_results[name])

    out = config.ANALYSIS_DIR / "figure7_capabilities.json"
    out.write_text(json.dumps(all_results, indent=2))
    print("saved:", out)


if __name__ == "__main__":
    main()
