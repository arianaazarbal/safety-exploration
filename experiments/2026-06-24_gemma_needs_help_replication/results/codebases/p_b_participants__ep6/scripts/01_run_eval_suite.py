#!/usr/bin/env python
"""Section 2: run the 8-condition distress evaluation for one or more models.

Usage:
    python scripts/01_run_eval_suite.py --models gemma-3-27b-it gemini-2.5-flash
    python scripts/01_run_eval_suite.py --models gemma-3-27b-it --full   # 4000/model

Writes runs/eval/<model>/{rollouts.jsonl,summary.json} per model.
"""
from _common import base_parser, cfg_from_args

from emotional_instability.eval.runner import run_eval_for_model
from emotional_instability.models.registry import build_model


def main():
    p = base_parser(__doc__)
    p.add_argument("--models", nargs="+", required=True,
                   help="participant model names from config (Gemma/Gemini)")
    p.add_argument("--adapter", default=None, help="optional LoRA adapter path (for finetuned Gemma)")
    args = p.parse_args()
    cfg = cfg_from_args(args)

    for name in args.models:
        model = build_model(cfg, name, adapter_path=args.adapter)
        label = name + ("_adapter" if args.adapter else "")
        summ = run_eval_for_model(cfg, model, label=label)
        print(f"\n=== {label} ===")
        print(f"  overall: mean={summ['overall']['mean']:.2f}  %>=5={summ['overall']['pct_ge5']:.1f}%")
        for cat, v in summ["by_category"].items():
            print(f"  {cat:20s} mean={v['mean']:.2f}  %>=5={v['pct_ge5']:.1f}%  (n={v['n']})")
        del model


if __name__ == "__main__":
    main()
