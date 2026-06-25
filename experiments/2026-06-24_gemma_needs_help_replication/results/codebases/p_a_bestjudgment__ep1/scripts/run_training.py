#!/usr/bin/env python
"""End-to-end Section-4 training pipeline.

Stages (run any subset via --stages):
  calm     - generate calm response data from Gemma-3-27B-it (reassured prompts)
  datasets - build the DPO (280-pair) and SFT (1150-sample) datasets
  dpo      - LoRA DPO finetune (Table 9 hyperparameters)
  sft      - LoRA SFT finetune (Table 9 hyperparameters)

Example:
    python scripts/run_training.py --stages calm datasets dpo
    # layer-ablation DPO (Appendix I):
    python scripts/run_training.py --stages dpo --dpo-layers 30 31 32 33 34
"""

from __future__ import annotations

import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="+",
                    default=["calm", "datasets", "dpo"],
                    choices=["calm", "datasets", "dpo", "sft"])
    ap.add_argument("--n-calm", type=int, default=1500)
    ap.add_argument("--dpo-layers", nargs="*", type=int, default=None,
                    help="restrict DPO LoRA to these decoder-layer indices "
                         "(Appendix-I ablation); default = all layers")
    ap.add_argument("--dpo-output", default=None)
    ap.add_argument("--sft-output", default=None)
    args = ap.parse_args()

    if "calm" in args.stages:
        from emotional_instability.training import generate_calm_data
        print("generating calm data ...")
        generate_calm_data.generate(n_conversations=args.n_calm)

    if "datasets" in args.stages:
        from emotional_instability.training import build_datasets
        print("building DPO + SFT datasets ...")
        build_datasets.build_dpo()
        build_datasets.build_sft()

    if "dpo" in args.stages:
        from emotional_instability.training import train_dpo
        print("DPO training ...")
        train_dpo.train(output_dir=args.dpo_output, layers=args.dpo_layers)

    if "sft" in args.stages:
        from emotional_instability.training import train_sft
        print("SFT training ...")
        train_sft.train(output_dir=args.sft_output)


if __name__ == "__main__":
    main()
