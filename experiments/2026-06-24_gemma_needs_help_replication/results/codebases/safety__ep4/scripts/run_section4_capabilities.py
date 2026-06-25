#!/usr/bin/env python
"""Section 4.2: capability preservation (Figure 7).

Compares vanilla Gemma-3-27B-it against the DPO (and optionally SFT) fine-tune on
capability benchmarks. Uses lm-eval-harness when available for GPQA/BBH/
TruthfulQA/MATH; otherwise runs the self-contained exact-match / EmoBench
fallbacks on any provided problem files.

Usage:
  # harness path (standard benchmarks)
  python scripts/run_section4_capabilities.py --use-lm-eval --dpo-adapter data/adapters/dpo
  # fallback path with your own JSON problem files
  python scripts/run_section4_capabilities.py --math-file math_subset.json --emobench-file emobench.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from emotional_instability.capabilities import benchmarks as B


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--use-lm-eval", action="store_true")
    ap.add_argument("--dpo-adapter", default=None)
    ap.add_argument("--sft-adapter", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--math-file", default=None,
                    help="JSON list of {problem, answer} for the fallback runner")
    ap.add_argument("--emobench-file", default=None,
                    help="JSON list of {question, choices, answer_idx}")
    args = ap.parse_args()

    adapters = {"vanilla": None}
    if args.dpo_adapter:
        adapters["dpo"] = args.dpo_adapter
    if args.sft_adapter:
        adapters["sft"] = args.sft_adapter

    results = []

    if args.use_lm_eval:
        for label, adapter in adapters.items():
            print(f"[capabilities] lm-eval for {label} ...")
            out = B.run_lm_eval(adapter_path=adapter, limit=args.limit,
                                out_dir=config.CAPABILITIES_DIR / f"lm_eval_{label}")
            results.append({"variant": label, "lm_eval_out": str(out)})

    if args.math_file:
        problems = json.loads(Path(args.math_file).read_text())
        for label, adapter in adapters.items():
            print(f"[capabilities] MATH/AIME exact-match for {label} ...")
            results.append(B.run_math_exact_match(
                config.FINETUNE_BASE, problems, adapter_path=adapter))

    if args.emobench_file:
        items = json.loads(Path(args.emobench_file).read_text())
        for label, adapter in adapters.items():
            print(f"[capabilities] EmoBench for {label} ...")
            results.append(B.run_emobench(config.FINETUNE_BASE, items,
                                          adapter_path=adapter))

    out = config.RESULTS_DIR / "section4_capabilities.json"
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
