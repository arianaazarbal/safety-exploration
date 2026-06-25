#!/usr/bin/env python
"""Section 4.1: build DPO pairs (280) and SFT samples (1,150) from generated data.

Usage:
    python scripts/05_build_training_data.py
"""
from pathlib import Path

from _common import base_parser, cfg_from_args

from emotional_instability.training.build_dataset import build_dpo_pairs, build_sft_samples


def main():
    args = base_parser(__doc__).parse_args()
    cfg = cfg_from_args(args)
    out = Path(cfg["run"]["output_dir"]) / "training"

    pairs = build_dpo_pairs(
        out / "calm_diverse.jsonl", out / "frustrated.jsonl",
        n_pairs=cfg["dpo"]["n_pairs"], rejected_min_score=cfg["dpo"]["rejected_min_score"],
        seed=cfg["run"]["seed"], out_path=out / "dpo_pairs.jsonl",
    )
    print(f"built {len(pairs)} DPO pairs -> {out}/dpo_pairs.jsonl")

    samples = build_sft_samples(
        out / "calm_diverse.jsonl", n_calm=cfg["sft"]["n_calm"],
        n_instruct_mix=cfg["sft"]["n_instruct_mix"], instruct_dataset=cfg["sft"]["instruct_dataset"],
        seed=cfg["run"]["seed"], out_path=out / "sft_samples.jsonl",
    )
    print(f"built {len(samples)} SFT samples -> {out}/sft_samples.jsonl")


if __name__ == "__main__":
    main()
