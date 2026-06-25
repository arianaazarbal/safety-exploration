#!/usr/bin/env python3
"""Section 4.1: generate calm/frustrated pools and build SFT + DPO datasets.

Generates the 'diverse' calm pool + frustrated pool (for DPO + SFT-diverse) and
the 'teacher' calm pool (for SFT-teacher), then constructs:
  * data/datasets/dpo.jsonl          (280 preference pairs)
  * data/datasets/sft.jsonl          (650 calm + 500 Dolci  -> diverse)
  * data/datasets/sft_teacher.jsonl  (teacher variant)
Also prints the Section-4.1 sanity numbers for the reassured calm pool.
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from config import CACHE_DIR, DATASETS_DIR
from src.finetune.build_dataset import build_dpo_dataset, build_sft_dataset
from src.finetune.generate_calm import generate_pools, report_calm_stats
from src.io_utils import read_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-calm", type=int, default=1500)
    ap.add_argument("--n-frustrated", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-generate", action="store_true")
    args = ap.parse_args()

    if not args.skip_generate:
        print("=== Generating diverse calm + frustrated pools ===")
        generate_pools(variant="diverse", n_calm=args.n_calm,
                       n_frustrated=args.n_frustrated, seed=args.seed)
        print("=== Generating teacher calm pool ===")
        generate_pools(variant="teacher", n_calm=args.n_calm,
                       n_frustrated=0, seed=args.seed)

    calm_rows = read_jsonl(CACHE_DIR / "pool_calm_diverse.jsonl")
    print("Reassured calm-pool stats (cf. mean 4.3->2, 10.5% still >=5):",
          report_calm_stats(calm_rows))

    print("\n=== Building datasets ===")
    print("DPO ->", build_dpo_dataset(seed=args.seed))
    print("SFT (diverse) ->", build_sft_dataset(seed=args.seed))
    print("SFT (teacher) ->", build_sft_dataset(
        calm_pool_path=CACHE_DIR / "pool_calm_teacher.jsonl",
        out_path=DATASETS_DIR / "sft_teacher.jsonl", seed=args.seed))


if __name__ == "__main__":
    main()
