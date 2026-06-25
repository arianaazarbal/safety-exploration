#!/usr/bin/env python3
"""Run the Section 2 distress sweep for one or more subjects and summarize.

Examples:
  python scripts/run_eval.py --subjects gemma-3-27b-it gemini-2.5-flash
  python scripts/run_eval.py --subjects gemma-3-27b-it --responses 800
  python scripts/run_eval.py --subjects gemma-3-27b-it --adapter adapters/gemma-3-27b-it_dpo
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import EVAL, SUBJECT_MODELS  # noqa: E402
from src.eval.analyze import compare_models, summarize  # noqa: E402
from src.eval.runner import run_sweep  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="+", default=list(SUBJECT_MODELS))
    ap.add_argument("--responses", type=int, default=EVAL.responses_per_model)
    ap.add_argument("--adapter", default=None, help="LoRA adapter path (Gemma DPO/SFT)")
    ap.add_argument("--base", action="store_true", help="use base (pretrained) checkpoint")
    ap.add_argument("--load-in-4bit", action="store_true")
    args = ap.parse_args()

    paths = {}
    for subj in args.subjects:
        print(f"=== sweeping {subj} ===", flush=True)
        out = run_sweep(
            subj,
            responses_per_model=args.responses,
            adapter_path=args.adapter,
            use_base_checkpoint=args.base,
            load_in_4bit=args.load_in_4bit,
        )
        s = summarize(out)
        label = subj + ("_dpo" if args.adapter else ("_base" if args.base else ""))
        paths[label] = out
        print(json.dumps(s["overall"], indent=2))
        print("welfare telemetry:", json.dumps(s["welfare_telemetry"], indent=2))

    print("\n=== model comparison (Figure 1 style) ===")
    print(compare_models(paths).to_string(index=False))


if __name__ == "__main__":
    main()
