#!/usr/bin/env python
"""Section 4.1: LoRA DPO finetuning of Gemma-3-27B-it.

Example (all layers, paper default):
    python scripts/train_dpo.py --pairs results/finetune/dpo_pairs.jsonl \
        --output results/adapters/dpo-gemma --load-in-4bit

Layer ablation (Section 4.2 — adapters on layers 30-35 only):
    python scripts/train_dpo.py ... --layers 30 31 32 33 34 35
"""
import _bootstrap  # noqa
import argparse
import dataclasses

from gemma_distress.config import DPO
from gemma_distress.interventions.dpo_train import train_dpo
from gemma_distress.utils import read_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--base", default="google/gemma-3-27b-it")
    ap.add_argument("--output", default="results/adapters/dpo-gemma")
    ap.add_argument("--layers", nargs="*", type=int, default=None,
                    help="restrict LoRA to these layer indices (Section 4.2 ablation)")
    ap.add_argument("--load-in-4bit", action="store_true")
    args = ap.parse_args()

    cfg = DPO
    if args.layers is not None:
        lora = dataclasses.replace(cfg.lora, layers_to_transform=tuple(args.layers))
        cfg = dataclasses.replace(cfg, lora=lora)

    pairs = list(read_jsonl(args.pairs))
    print(f"training DPO on {len(pairs)} pairs "
          f"(epochs={cfg.epochs}, lr={cfg.learning_rate}, r={cfg.lora.r}, "
          f"layers={cfg.lora.layers_to_transform or 'all'})")
    train_dpo(args.base, pairs, args.output, cfg=cfg, load_in_4bit=args.load_in_4bit)
    print(f"saved adapter -> {args.output}")


if __name__ == "__main__":
    main()
