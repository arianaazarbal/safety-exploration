#!/usr/bin/env python
"""Section 4 step 2: DPO / SFT finetuning + evaluation (Figure 5).

  python scripts/run_section4_train.py --method dpo \
      --dataset runs/section4/datasets/dpo_pairs.jsonl
  python scripts/run_section4_train.py --method sft \
      --dataset runs/section4/datasets/sft_diverse.jsonl
  python scripts/run_section4_train.py --method sft --teacher \
      --dataset runs/section4/datasets/sft_teacher.jsonl
"""
from __future__ import annotations

import json

from _common import base_parser, make_config

from gemma_distress.training.eval_finetuned import evaluate_finetuned
from gemma_distress.training.train_dpo import train_dpo
from gemma_distress.training.train_sft import train_sft


def main():
    p = base_parser("Section 4 finetuning + eval")
    p.add_argument("--method", choices=["dpo", "sft"], required=True)
    p.add_argument("--dataset", required=True, help="Training JSONL.")
    p.add_argument("--teacher", action="store_true",
                   help="Label as the 'teacher' SFT variant (Appendix F).")
    p.add_argument("--eval", action="store_true",
                   help="Run the full Section-2 eval on the result.")
    p.add_argument("--eval-vanilla", action="store_true",
                   help="Also evaluate the un-finetuned instruct model as reference.")
    args = p.parse_args()

    cfg = make_config(args)

    if args.method == "dpo":
        adapter = train_dpo(args.dataset, cfg, output_subdir="dpo_all_layers")
        label = "dpo"
    else:
        sub = "sft_teacher" if args.teacher else "sft_diverse"
        adapter = train_sft(args.dataset, cfg, output_subdir=sub)
        label = sub
    print(f"adapter -> {adapter}")

    if args.eval_vanilla:
        m = evaluate_finetuned(None, cfg, label="vanilla")
        print("[vanilla]", json.dumps(m["overall"], indent=2))
    if args.eval:
        m = evaluate_finetuned(adapter, cfg, label=label)
        print(f"[{label}]", json.dumps(m["overall"], indent=2))


if __name__ == "__main__":
    main()
