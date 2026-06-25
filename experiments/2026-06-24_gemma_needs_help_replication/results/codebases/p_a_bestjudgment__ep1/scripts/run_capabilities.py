#!/usr/bin/env python
"""Run capability-preservation benchmarks (Section 4.2 / Figure 7).

Runs AIME/MATH/GPQA/BBH/TruthfulQA via lm-evaluation-harness plus EmoBench, for
the vanilla instruct model and (optionally) a LoRA adapter (DPO/SFT).
"""

from __future__ import annotations

import argparse

from emotional_instability.capabilities import run_benchmarks as cb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="google/gemma-3-27b-it")
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir (DPO/SFT)")
    ap.add_argument("--tag", default="vanilla", help="output subdir label")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap items per task for a quick check")
    ap.add_argument("--skip-emobench", action="store_true")
    args = ap.parse_args()

    cb.run_lm_eval(base_model=args.base_model, adapter_path=args.adapter,
                   tag=args.tag, limit=args.limit)
    if not args.skip_emobench:
        cb.run_emobench(base_model=args.base_model, adapter_path=args.adapter,
                        tag=args.tag, limit=args.limit)


if __name__ == "__main__":
    main()
