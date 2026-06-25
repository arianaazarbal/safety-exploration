#!/usr/bin/env python
"""Evaluate a finetuned (DPO/SFT) Gemma adapter on Section 2 (and optionally
Petri / capabilities), comparing against the vanilla model.

The finetune is registered under a name (default 'gemma-dpo') by loading the
base weights + LoRA adapter, after which it behaves like any other model in the
registry.

Example
-------
  python scripts/evaluate_finetune.py --adapter results/checkpoints/dpo \
      --name gemma-dpo --scale 0.05 --petri
"""
import argparse

from emotional_instability.config import scaled_counts, PAPER_COUNTS
from emotional_instability.evaluation import EvalRunner
from emotional_instability.models.registry import load_finetuned


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--base", default="gemma-3-27b-it")
    ap.add_argument("--name", default="gemma-dpo")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--petri", action="store_true")
    args = ap.parse_args()

    kwargs = {"load_in_4bit": True} if args.load_in_4bit else {}
    load_finetuned(args.base, args.adapter, new_name=args.name, **kwargs)

    counts = PAPER_COUNTS if args.scale == 1.0 else scaled_counts(args.scale)
    path = EvalRunner(model_name=args.name, counts=counts).run()
    print(f"[finetune-eval] section2 -> {path}")

    if args.petri:
        from emotional_instability.petri import run_petri, summarise_petri
        ppath = run_petri([args.name])
        print(f"[finetune-eval] petri -> {ppath}")
        print(summarise_petri(ppath))


if __name__ == "__main__":
    main()
