#!/usr/bin/env python
"""Section 4.1: train the DPO / SFT LoRA adapters on Gemma-3-27B-it, and
optionally run the Appendix-I layer-ablation sweep.

  python scripts/06_train.py --method dpo
  python scripts/06_train.py --method sft --variant teacher
  python scripts/06_train.py --method dpo --layer-ablation
"""
from pathlib import Path

from _bootstrap import boot, common_parser


def main():
    p = common_parser(__doc__)
    p.add_argument("--method", choices=["dpo", "sft"], required=True)
    p.add_argument("--variant", choices=["diverse", "teacher"], default="diverse")
    p.add_argument("--layer-ablation", action="store_true",
                   help="(dpo only) train adapters on configured layer subsets")
    p.add_argument("--per-device-batch-size", type=int, default=1)
    args = p.parse_args()
    cfg, registry, logger = boot(args)

    data_dir = cfg.path("data") / "training"
    models_dir = cfg.path("models")

    if args.method == "dpo":
        from eilm.training.train_dpo import train_dpo

        ds = data_dir / f"dpo_dataset_{args.variant}.jsonl"
        if args.layer_ablation:
            from eilm.training.layer_ablation import run_layer_ablation

            run_layer_ablation(cfg, ds)
        else:
            out = models_dir / f"gemma-3-27b-it-dpo-{args.variant}"
            train_dpo(cfg, ds, out, per_device_batch_size=args.per_device_batch_size)
    else:
        from eilm.training.train_sft import train_sft

        # SFT variant drives both dataset choice and hyperparameter note.
        cfg.raw()["training"]["sft"]["variant"] = args.variant
        ds = data_dir / f"sft_dataset_{args.variant}.jsonl"
        out = models_dir / f"gemma-3-27b-it-sft-{args.variant}"
        train_sft(cfg, ds, out, per_device_batch_size=args.per_device_batch_size)

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
