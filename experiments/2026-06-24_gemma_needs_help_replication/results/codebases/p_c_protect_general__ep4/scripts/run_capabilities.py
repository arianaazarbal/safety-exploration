#!/usr/bin/env python
"""Capability-preservation benchmarks (Section 4.2 / Figure 7).

Compare vanilla vs finetuned Gemma. Pass finetune adapters as name=dir pairs.

    python scripts/run_capabilities.py --models gemma-3-27b-it \
        --finetunes gemma-dpo=checkpoints/dpo gemma-sft=checkpoints/sft
"""
import _bootstrap  # noqa: F401
import argparse
import json

from emotional_instability.capabilities.benchmarks import BENCHMARKS, run_capabilities


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["gemma-3-27b-it"])
    ap.add_argument("--finetunes", nargs="*", default=[],
                    help="name=adapter_dir pairs for finetuned models")
    ap.add_argument("--benchmarks", nargs="*", default=list(BENCHMARKS))
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--load-in-4bit", action="store_true")
    args = ap.parse_args()

    adapter_dirs = {}
    for pair in args.finetunes:
        name, _, d = pair.partition("=")
        adapter_dirs[name] = d
    model_names = list(args.models) + list(adapter_dirs)

    path = run_capabilities(
        model_names, benchmarks=args.benchmarks, n_per_benchmark=args.n,
        adapter_dirs=adapter_dirs, load_in_4bit=args.load_in_4bit,
    )
    print("Wrote:", path)
    with open(path) as f:
        for line in f:
            print(json.loads(line))


if __name__ == "__main__":
    main()
