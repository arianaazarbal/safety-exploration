"""Section 4 data prep: generate calm responses, then build the SFT and DPO
datasets.

Example:
    python scripts/build_finetune_data.py --eval results/eval_gemma-3-27b-it_smoke.jsonl
"""
import _bootstrap  # noqa: F401
import argparse
from pathlib import Path

import config
from src.training.generate_calm import generate_calm_pool
from src.training.build_dataset import build_dpo_dataset, build_sft_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True, type=Path,
                    help="Section 2 results JSONL for gemma-3-27b-it (frustrated pool)")
    ap.add_argument("--skip-calm", action="store_true",
                    help="reuse an existing calm_pool.jsonl")
    args = ap.parse_args()

    calm_path = config.ARTIFACT_DIR / "calm_pool.jsonl"
    if not args.skip_calm:
        calm_path = generate_calm_pool()

    build_sft_dataset(calm_path)
    build_dpo_dataset(args.eval)


if __name__ == "__main__":
    main()
