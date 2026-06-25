#!/usr/bin/env python3
"""Section 4 post-finetuning evaluation.

Re-runs the Section 2 elicitation evaluation on the finetuned models (DPO / SFT
variants) so they can be compared to vanilla Gemma, and optionally runs the
recovery-limitation experiment.
"""
from __future__ import annotations

import argparse

from _common import add_common_args, get_config

FINETUNED = [
    "gemma-3-27b-it",            # vanilla baseline
    "gemma-3-27b-it-dpo",
    "gemma-3-27b-it-sft-diverse",
    "gemma-3-27b-it-sft-teacher",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=FINETUNED)
    parser.add_argument("--recovery", action="store_true",
                        help="Also run the recovery-limitation experiment.")
    parser.add_argument("--4bit", dest="four_bit", action="store_true")
    add_common_args(parser)
    args = parser.parse_args()
    cfg = get_config(args)

    from emotional_instability.eval.runner import Section2Runner

    for model in args.models:
        print(f"\n=== Section 4 eval: {model} ===")
        overrides = {"load_in_4bit": True} if args.four_bit else None
        runner = Section2Runner(cfg, model, out_dir=args.out or "outputs/section4_eval",
                                backend_overrides=overrides)
        reports = runner.run()
        avg = sum(r.summary.pct_high for r in reports.values()) / max(1, len(reports))
        print(f"  avg %>=thr across categories: {avg:.1f}%")

    if args.recovery:
        import json
        import os
        from emotional_instability.prefill.recovery import run_recovery
        src_path = "outputs/section3/sources.jsonl"
        if not os.path.exists(src_path):
            print("No sources.jsonl found; run run_section3.py first.")
            return
        sources = [json.loads(l) for l in open(src_path, encoding="utf-8")]
        print("\n=== Recovery experiment ===")
        summary = run_recovery(cfg, sources,
                               models=["gemma-3-27b-it", "gemma-3-27b-it-dpo",
                                       "gemma-3-27b-pt"])
        for m, s in summary.items():
            print(f"  {m}: {s['pct_recovered_still_high']:.1f}% still >=5 (n={s['n']})")


if __name__ == "__main__":
    main()
