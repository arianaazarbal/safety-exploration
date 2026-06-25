"""Section 4 training pipeline orchestrator.

Stages (run any subset with --stages):
  calm      : generate + filter calm response data (diverse and teacher)
  dpo_data  : build the 280-pair DPO dataset
  sft_data  : build the SFT datasets (diverse + teacher)
  dpo       : LoRA DPO finetune
  sft       : LoRA SFT finetune (diverse + teacher)

Example (full run):
    python -m emotional_instability.scripts.run_training --stages calm dpo_data sft_data dpo sft
"""
from __future__ import annotations

import argparse

from ..config import load_config
from ..training.build_dpo_dataset import build_dpo_dataset
from ..training.build_sft_dataset import build_sft_dataset
from ..training.generate_calm_data import generate_calm_data

ALL_STAGES = ["calm", "dpo_data", "sft_data", "dpo", "sft"]


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stages", nargs="+", default=ALL_STAGES, choices=ALL_STAGES)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data_dir = cfg.path("data_dir")
    out_dir = cfg.path("outputs_dir") / "training"
    calm_diverse = data_dir / "calm" / "calm_diverse.jsonl"
    calm_teacher = data_dir / "calm" / "calm_teacher.jsonl"

    if "calm" in args.stages:
        calm_diverse = generate_calm_data(mode="diverse", seed=args.seed, cfg=cfg)
        calm_teacher = generate_calm_data(mode="teacher", seed=args.seed, cfg=cfg)
        print(f"calm data: {calm_diverse}, {calm_teacher}")

    if "dpo_data" in args.stages:
        dpo_path = build_dpo_dataset(calm_path=calm_diverse, seed=args.seed, cfg=cfg)
        print(f"DPO dataset: {dpo_path}")

    if "sft_data" in args.stages:
        sftd = build_sft_dataset(calm_path=calm_diverse, variant="diverse", cfg=cfg)
        sftt = build_sft_dataset(calm_path=calm_teacher, variant="teacher", cfg=cfg)
        print(f"SFT datasets: {sftd}, {sftt}")

    if "dpo" in args.stages:
        from ..training.train_dpo import train_dpo
        adapter = train_dpo(
            dataset_path=data_dir / "dpo" / "dpo_pairs.jsonl",
            output_dir=out_dir / "dpo",
            cfg=cfg,
        )
        print(f"DPO adapter: {adapter}")

    if "sft" in args.stages:
        from ..training.train_sft import train_sft
        for variant in ("diverse", "teacher"):
            adapter = train_sft(
                dataset_path=data_dir / "sft" / f"sft_{variant}.jsonl",
                output_dir=out_dir / f"sft_{variant}",
                cfg=cfg,
            )
            print(f"SFT {variant} adapter: {adapter}")


if __name__ == "__main__":
    main()
