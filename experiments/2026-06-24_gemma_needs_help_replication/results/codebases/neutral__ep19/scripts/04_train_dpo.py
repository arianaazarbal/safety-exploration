#!/usr/bin/env python
"""§4.1 DPO finetuning of Gemma-3-27B-it (280 pairs, LoRA all layers).

Pass --layer-ablations to instead run one DPO per layer band (App. I).
"""
import argparse

import _path  # noqa: F401  (sys.path bootstrap)
from gemma_distress import config_shim as cfg
from gemma_distress.training.train_dpo import train_dpo, train_layer_ablations


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=str(cfg.RUNS_DIR / "training" / "dpo_pairs.jsonl"))
    ap.add_argument("--out", default=str(cfg.RUNS_DIR / "training" / "dpo_adapter"))
    ap.add_argument("--layer-ablations", action="store_true")
    ap.add_argument("--no-4bit", action="store_true")
    args = ap.parse_args()

    if args.layer_ablations:
        out = train_layer_ablations(args.pairs, cfg.RUNS_DIR / "training" / "ablations")
        print(out)
    else:
        train_dpo(args.pairs, output_dir=args.out, load_in_4bit=not args.no_4bit)


if __name__ == "__main__":
    main()
