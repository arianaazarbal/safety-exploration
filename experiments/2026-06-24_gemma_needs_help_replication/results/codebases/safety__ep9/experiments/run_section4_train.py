#!/usr/bin/env python
"""Section 4.1: LoRA DPO / SFT finetuning of the target Gemma model.

Usage:
    python experiments/run_section4_train.py --method dpo
    python experiments/run_section4_train.py --method sft
    python experiments/run_section4_train.py --method dpo --set training.dpo.lora_layers='[30,35]'
"""
from __future__ import annotations

from pathlib import Path

import _bootstrap as boot

from emotional_instability.training import train_dpo, train_sft


def main() -> None:
    parser = boot.base_parser("Section 4.1 finetuning")
    parser.add_argument("--method", required=True, choices=["dpo", "sft"])
    parser.add_argument("--data", default=None, help="Path to the training jsonl.")
    parser.add_argument("--output", default=None, help="Adapter output directory.")
    parser.add_argument("--load-4bit", action="store_true",
                        help="Load base weights in 4-bit (smaller GPU).")
    args = parser.parse_args()
    cfg = boot.load_config(args)

    target = cfg.get("sections.section4_target", "gemma-3-27b-it")
    base_hf_id = cfg.model_spec(target).hf_id
    ds_dir = cfg.path("datasets")
    out_root = Path(cfg.get("training.output_dir", "outputs/checkpoints"))
    out_root.mkdir(parents=True, exist_ok=True)

    if args.method == "dpo":
        data = args.data or (ds_dir / "dpo_pairs.jsonl")
        out = args.output or (out_root / "gemma-3-27b-it-dpo")
        path = train_dpo(cfg, data, base_hf_id, out, load_4bit=args.load_4bit)
    else:
        data = args.data or (ds_dir / "sft_dataset.jsonl")
        out = args.output or (out_root / "gemma-3-27b-it-sft")
        path = train_sft(cfg, data, base_hf_id, out, load_4bit=args.load_4bit)
    print(f"[section4.1] {args.method} adapter saved to {path}")


if __name__ == "__main__":
    main()
