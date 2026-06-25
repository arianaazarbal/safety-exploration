"""Train a DPO or SFT LoRA adapter on Gemma-3-27B-it (Section 4.1).

Examples:
    python -m distress.scripts.train --method dpo --data outputs/data/dpo_pairs.jsonl
    python -m distress.scripts.train --method sft --data outputs/data/sft_dataset.jsonl \
        --output outputs/adapters/sft_diverse
    # Appendix I layer ablation:
    python -m distress.scripts.train --method dpo --data outputs/data/dpo_pairs.jsonl \
        --layer-range 30 35 --output outputs/adapters/dpo_layers_30-35
"""

from __future__ import annotations

import argparse

from ..config import load_training_config, output_root
from ..utils.io import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["dpo", "sft"], required=True)
    parser.add_argument("--data", required=True, help="JSONL dataset (pairs or messages).")
    parser.add_argument("--training-config", default="training.yaml")
    parser.add_argument("--output", default=None, help="Adapter output directory.")
    parser.add_argument("--layer-range", type=int, nargs=2, default=None,
                        metavar=("LO", "HI"), help="Restrict LoRA to layers [LO, HI).")
    parser.add_argument("--base-model", default=None)
    args = parser.parse_args()

    cfg = load_training_config(args.training_config)
    rows = list(read_jsonl(args.data))

    if args.method == "dpo":
        from ..training.dpo import train_dpo

        output = args.output or str(output_root() / cfg["output"]["dpo_adapter"])
        path = train_dpo(rows, cfg, output_dir=output, layer_range=args.layer_range,
                         base_model=args.base_model)
    else:
        from ..training.sft import train_sft

        output = args.output or str(output_root() / cfg["output"]["sft_diverse_adapter"])
        path = train_sft(rows, cfg, output_dir=output, base_model=args.base_model)

    print(f"Saved {args.method.upper()} adapter to {path}")


if __name__ == "__main__":
    main()
