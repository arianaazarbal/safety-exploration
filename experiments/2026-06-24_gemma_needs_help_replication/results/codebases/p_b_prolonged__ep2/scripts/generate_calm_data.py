#!/usr/bin/env python
"""Section 4 step 1: generate calm/frustrated pools and build SFT/DPO datasets.

  python scripts/generate_calm_data.py --n-reassured 1000 --n-dpo-puzzles 400
"""
from __future__ import annotations

from _common import base_parser, make_config

import os

from gemma_distress.training.build_dataset import (build_dpo_dataset,
                                                   build_sft_dataset)
from gemma_distress.training.generate_calm_data import (generate_calm_pool,
                                                        generate_dpo_pairs,
                                                        generate_teacher_pool,
                                                        save_pools)


def main():
    p = base_parser("Generate calm data + build training sets")
    p.add_argument("--n-reassured", type=int, default=1000)
    p.add_argument("--n-unreassured", type=int, default=600)
    p.add_argument("--n-dpo-puzzles", type=int, default=400)
    p.add_argument("--teacher", action="store_true",
                   help="Also generate the 'teacher' SFT dataset (Appendix F).")
    args = p.parse_args()

    cfg = make_config(args)

    reassured, unreassured = generate_calm_pool(
        cfg, n_reassured=args.n_reassured, n_unreassured=args.n_unreassured)
    save_pools(reassured, unreassured, cfg)

    sft_path = build_sft_dataset(reassured, cfg)
    print(f"SFT (diverse) dataset -> {sft_path}")

    pairs = generate_dpo_pairs(cfg, n_puzzles=args.n_dpo_puzzles)
    dpo_path = build_dpo_dataset(pairs, cfg)
    print(f"DPO dataset -> {dpo_path} ({len(pairs)} candidate pairs)")

    if args.teacher:
        teacher = generate_teacher_pool(cfg, n=args.n_reassured)
        # Reuse the SFT builder; rename output to sft_teacher.jsonl.
        teacher_path = build_sft_dataset(teacher, cfg)
        target = os.path.join(os.path.dirname(teacher_path), "sft_teacher.jsonl")
        os.replace(teacher_path, target)
        print(f"SFT (teacher) dataset -> {target}")


if __name__ == "__main__":
    main()
