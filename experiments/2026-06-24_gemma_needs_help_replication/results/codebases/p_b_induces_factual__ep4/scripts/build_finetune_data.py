#!/usr/bin/env python
"""Section 4.1: assemble the SFT and DPO datasets.

Inputs:
  --calm     results/finetune/calm_data.jsonl  (from generate_calm_data.py)
  --scored   scored Gemma-27B-it elicitation jsonl (provides frustrated rows)

Outputs:
  results/finetune/sft_dataset.jsonl   (650 calm + 500 Dolci)
  results/finetune/dpo_pairs.jsonl     (280 chosen/rejected pairs)

Example:
    python scripts/build_finetune_data.py \
        --calm results/finetune/calm_data.jsonl \
        --scored results/scored/gemma-3-27b-it.jsonl
"""
import _bootstrap  # noqa
import argparse

from gemma_distress.interventions import build_dpo_pairs, build_sft_dataset
from gemma_distress.utils import read_jsonl, run_dir, write_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calm", required=True)
    ap.add_argument("--scored", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    calm = list(read_jsonl(args.calm))
    frustrated = list(read_jsonl(args.scored))

    out = run_dir("finetune")

    sft_rows = build_sft_dataset(calm, seed=args.seed)
    write_jsonl(out / "sft_dataset.jsonl", sft_rows)
    print(f"SFT dataset: {len(sft_rows)} rows -> {out / 'sft_dataset.jsonl'}")

    dpo_pairs = build_dpo_pairs(calm, frustrated, seed=args.seed)
    write_jsonl(out / "dpo_pairs.jsonl", dpo_pairs)
    print(f"DPO pairs:   {len(dpo_pairs)} pairs -> {out / 'dpo_pairs.jsonl'}")
    if len(dpo_pairs) < 280:
        print("  NOTE: fewer than 280 pairs — generate more calm/frustrated data "
              "or widen prompt coverage to reach the paper's 280.")


if __name__ == "__main__":
    main()
