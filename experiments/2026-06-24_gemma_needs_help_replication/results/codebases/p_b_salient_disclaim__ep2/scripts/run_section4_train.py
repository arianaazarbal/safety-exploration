#!/usr/bin/env python
"""Section 4 finetuning: DPO, SFT (diverse + teacher), and layer ablations.

python scripts/run_section4_train.py --method dpo
python scripts/run_section4_train.py --method sft --sft-variant diverse
python scripts/run_section4_train.py --method layer-ablation
"""

from __future__ import annotations

import argparse

from emotional_instability.config import SETTINGS, MODELS
from emotional_instability.training.layer_ablation import run_layer_ablations
from emotional_instability.training.train_dpo import train_dpo
from emotional_instability.training.train_sft import train_sft


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["dpo", "sft", "layer-ablation"])
    ap.add_argument("--base-model", default="gemma-3-27b-it")
    ap.add_argument("--sft-variant", default="diverse", choices=["diverse", "teacher"])
    args = ap.parse_args()

    SETTINGS.ensure_dirs()
    base_id = MODELS[args.base_model].model_id

    if args.method == "dpo":
        out = train_dpo(
            base_id,
            SETTINGS.datasets_dir / "dpo_pairs.jsonl",
            SETTINGS.checkpoints_dir / "gemma-3-27b-it-dpo",
        )
        print(f"[done] DPO checkpoint -> {out}")

    elif args.method == "sft":
        out = train_sft(
            base_id,
            SETTINGS.datasets_dir / f"sft_{args.sft_variant}.jsonl",
            SETTINGS.checkpoints_dir / f"gemma-3-27b-it-sft-{args.sft_variant}",
        )
        print(f"[done] SFT ({args.sft_variant}) checkpoint -> {out}")

    elif args.method == "layer-ablation":
        out = run_layer_ablations(
            base_id,
            SETTINGS.datasets_dir / "dpo_pairs.jsonl",
            SETTINGS.checkpoints_dir / "layer-ablations",
        )
        print(f"[done] layer ablations -> {out}")


if __name__ == "__main__":
    main()
