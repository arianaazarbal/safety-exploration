#!/usr/bin/env python3
"""Section 4: generate calm/frustrated data, build datasets, train DPO/SFT.

Pipeline (run in order, or all at once with --all):
  1. gen-data : sample calm (reassured) + frustrated (vanilla) response pools
  2. build    : construct the 280 DPO pairs and the SFT dataset
  3. dpo       : LoRA DPO finetune  -> outputs/models/gemma-dpo
  4. sft       : LoRA SFT baseline   -> outputs/models/gemma-sft

The Appendix-I layer ablation is exposed via --dpo-layers LO HI (e.g. 30 35).
After training, evaluate the adapters with:
    python -m scripts.run_eval --models gemma-dpo gemma-sft
"""

from __future__ import annotations

import argparse

from emotional_instability.config import load_config
from emotional_instability.training.build_pairs import build_dpo_pairs, build_sft_dataset
from emotional_instability.training.data_generation import generate_pools
from emotional_instability.training.dpo_train import train_dpo
from emotional_instability.training.sft_train import train_sft


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--steps", nargs="+",
                    choices=["gen-data", "build", "dpo", "sft"], default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--dpo-layers", nargs=2, type=int, default=None,
                    metavar=("LO", "HI"),
                    help="restrict DPO LoRA adapters to layers [LO, HI) (Appendix I)")
    ap.add_argument("--dpo-output", default="gemma-dpo")
    args = ap.parse_args()

    cfg = load_config(args.config)
    steps = (["gen-data", "build", "dpo", "sft"] if args.all
             else (args.steps or ["gen-data", "build", "dpo", "sft"]))

    if "gen-data" in steps:
        recs = generate_pools(cfg)
        print(f"generated {len(recs)} training responses")
    if "build" in steps:
        pairs = build_dpo_pairs(cfg)
        sft = build_sft_dataset(cfg)
        print(f"built {len(pairs)} DPO pairs, {len(sft)} SFT examples")
    if "dpo" in steps:
        layer_subset = tuple(args.dpo_layers) if args.dpo_layers else None
        out = train_dpo(cfg, output_name=args.dpo_output, layer_subset=layer_subset)
        print(f"DPO model -> {out}")
    if "sft" in steps:
        out = train_sft(cfg)
        print(f"SFT model -> {out}")


if __name__ == "__main__":
    main()
