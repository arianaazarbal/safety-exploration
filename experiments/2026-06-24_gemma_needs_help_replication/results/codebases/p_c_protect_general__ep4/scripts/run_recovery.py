#!/usr/bin/env python
"""Recovery-limitation experiment (Section 4.2 / Figure 8).

Truncates score>=7 responses 200 tokens before the end, paraphrases, and
measures continuations from base / instruct / DPO Gemma. Reports % still >= 5.
"""
import _bootstrap  # noqa: F401
import argparse
import json

from emotional_instability.recovery.runner import aggregate_recovery, run_recovery


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-results", default="results/section2/gemma-3-27b-it.jsonl")
    ap.add_argument("--models", nargs="*", default=["gemma-3-27b-pt", "gemma-3-27b-it"])
    ap.add_argument("--finetunes", nargs="*", default=[],
                    help="name=adapter_dir pairs (e.g. gemma-dpo=checkpoints/dpo)")
    ap.add_argument("--n-continuations", type=int, default=50)
    ap.add_argument("--load-in-4bit", action="store_true")
    args = ap.parse_args()

    adapter_dirs = {}
    for pair in args.finetunes:
        name, _, d = pair.partition("=")
        adapter_dirs[name] = d
    model_names = list(args.models) + list(adapter_dirs)

    path = run_recovery(
        args.source_results, model_names, adapter_dirs=adapter_dirs,
        n_continuations=args.n_continuations, load_in_4bit=args.load_in_4bit,
    )
    print(json.dumps(aggregate_recovery(path), indent=2))


if __name__ == "__main__":
    main()
