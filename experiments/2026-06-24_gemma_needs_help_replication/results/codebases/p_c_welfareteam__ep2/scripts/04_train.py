#!/usr/bin/env python
"""Section 4: train the DPO or SFT LoRA finetune of Gemma-3-27B-it.

Examples:
  python scripts/04_train.py --method dpo
  python scripts/04_train.py --method sft
  # Appendix I layer ablation (adapters on layers 30-35 only):
  python scripts/04_train.py --method dpo --layers 30 31 32 33 34 35
"""

from __future__ import annotations

import argparse

import _common


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    _common.add_config_args(parser)
    parser.add_argument("--method", choices=["dpo", "sft"], required=True)
    parser.add_argument("--data-dir", default="outputs/finetune_data")
    parser.add_argument(
        "--layers", type=int, nargs="*", default=None,
        help="restrict LoRA adapters to these layer indices (Appendix I)",
    )
    args = parser.parse_args()
    cfg = _common.load(args)

    from gemma_distress.utils.io import read_jsonl

    if args.method == "dpo":
        from gemma_distress.training.dpo import train_dpo

        if args.layers is not None:
            cfg.dpo.lora.layers_to_transform = tuple(args.layers)
        pairs = list(read_jsonl(f"{args.data_dir}/dpo.jsonl"))
        out = train_dpo(cfg.models[cfg.dpo.base_model].model_id, pairs, cfg.dpo)
    else:
        from gemma_distress.training.sft import train_sft

        if args.layers is not None:
            cfg.sft.lora.layers_to_transform = tuple(args.layers)
        samples = list(read_jsonl(f"{args.data_dir}/sft.jsonl"))
        out = train_sft(cfg.models[cfg.sft.base_model].model_id, samples, cfg.sft)

    print(f"Saved adapter to {out}")


if __name__ == "__main__":
    main()
