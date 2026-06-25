#!/usr/bin/env python3
"""Section 4.1-4.2: train the SFT or DPO LoRA adapter on Gemma-3-27B-it.

Supports the Section 4.2 layer ablations via ``--ablation`` (all | layers_30_35
| layers_40_plus), which restricts the LoRA target layers to probe whether the
intervention acts on early/central layers.

Examples:
    python scripts/train.py --method dpo
    python scripts/train.py --method dpo --ablation layers_30_35
    python scripts/train.py --method sft
"""

from __future__ import annotations

import argparse

from _common import DATA_DIR, setup

from emotional_instability.training.dpo import train_dpo
from emotional_instability.training.sft import train_sft
from emotional_instability.utils.io import load_jsonl


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--method", choices=["sft", "dpo"], required=True)
    ap.add_argument("--base-model", default="gemma-3-27b-it")
    ap.add_argument("--ablation", default="all",
                    help="LoRA layer ablation key (see config finetuning.lora.layer_ablations).")
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    cfg = setup()
    ft = cfg.experiment["finetuning"]
    base_hf_id = cfg.target(args.base_model).identifier
    out_dir = args.output_dir or str(DATA_DIR / f"adapter_{args.method}_{args.ablation}")

    if args.method == "sft":
        records = load_jsonl(DATA_DIR / "sft_dataset.jsonl")
        # SFT records were written calm-first then Dolci; both are {"messages": ...}.
        sft_records = [r for r in records]
        path = train_sft(
            base_hf_id, sft_records, dolci_records=[],
            lora_cfg=ft["lora"], sft_cfg=ft["sft"],
            output_dir=out_dir, ablation=args.ablation, seed=cfg.seed,
        )
    else:
        dpo_records = load_jsonl(DATA_DIR / "dpo_dataset.jsonl")
        path = train_dpo(
            base_hf_id, dpo_records,
            lora_cfg=ft["lora"], dpo_cfg=ft["dpo"],
            output_dir=out_dir, ablation=args.ablation, seed=cfg.seed,
        )
    print(f"[done] {args.method} ({args.ablation}) adapter -> {path}")


if __name__ == "__main__":
    main()
