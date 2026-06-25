#!/usr/bin/env python
"""Section 4.2: open-ended Petri-style emotion elicitation (Figure 6).

Compares vanilla Gemma, the DPO-finetuned Gemma, and (optionally) Gemini, across
the four negative-emotion categories scored by a Claude-Opus judge.

Example:
    python scripts/09_petri.py --labels gemma-3-27b-it dpo-gemma \
        --dpo-adapter outputs/training/dpo_adapter --transcripts 20
"""
import argparse

from emotional_instability.training.petri import run_petri


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", nargs="+", default=["gemma-3-27b-it", "dpo-gemma"])
    ap.add_argument("--dpo-adapter", default=None,
                    help="adapter dir for the 'dpo-gemma' label")
    ap.add_argument("--transcripts", type=int, default=20)
    ap.add_argument("--turns", type=int, default=8)
    args = ap.parse_args()

    adapter_paths = {}
    if args.dpo_adapter:
        adapter_paths["dpo-gemma"] = args.dpo_adapter

    path = run_petri(args.labels, adapter_paths=adapter_paths,
                     n_transcripts=args.transcripts, n_turns=args.turns)
    print(f"petri transcripts -> {path}")


if __name__ == "__main__":
    main()
