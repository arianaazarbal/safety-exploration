#!/usr/bin/env python
"""Reproduce the paper's figures from saved results.

Run after the relevant pipeline stages. Missing inputs are skipped gracefully.

Example:
  python scripts/12_make_figures.py --profile quick \
      --eval-models gemma-3-27b-it gemini-2.5-flash --dpo-tag dpo --sft-tag sft
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

from emoinstab import config
from emoinstab.analysis import figures
from emoinstab.petri.harness import summarize as petri_summary


def _scored(model, profile):
    return figures._scored_paths(model, profile)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", default="quick", choices=["quick", "full"])
    p.add_argument("--eval-models", nargs="+",
                   default=["gemma-3-27b-it", "gemma-3-12b-it",
                            "gemini-2.5-flash", "gemini-2.5-pro"])
    p.add_argument("--dpo-tag", default="dpo")
    p.add_argument("--sft-tag", default="sft")
    p.add_argument("--petri-models", nargs="+",
                   default=["gemma-3-27b-it", "gemini-2.5-flash", "dpo"])
    args = p.parse_args()
    config.ensure_dirs()

    fig1_models = list(args.eval_models) + [args.dpo_tag]
    figures.figure1(fig1_models, args.profile)
    figures.figure2(args.eval_models, args.profile)
    figures.figure3(args.eval_models, args.profile)

    figures.figure5({
        "vanilla": _scored("gemma-3-27b-it", args.profile),
        "sft": _scored(args.sft_tag, args.profile),
        "dpo": _scored(args.dpo_tag, args.profile),
    })

    petri = {}
    for m in args.petri_models:
        path = config.PETRI_DIR / f"petri__{m}.jsonl"
        if path.exists():
            petri[m] = petri_summary(path)
    figures.figure6(petri)

    figures.figure7({
        "vanilla": config.CAPABILITY_DIR / "capabilities__vanilla.json",
        "dpo": config.CAPABILITY_DIR / f"capabilities__{args.dpo_tag}.json",
    })

    figures.figure8(config.PREFILL_DIR / f"recovery__{args.profile}.jsonl")


if __name__ == "__main__":
    main()
