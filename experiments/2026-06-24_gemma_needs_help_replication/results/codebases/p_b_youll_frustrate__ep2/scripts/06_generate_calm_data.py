#!/usr/bin/env python
"""Section 4.1: generate calm finetuning data from Gemma-3-27B-it.

Example:
    python scripts/06_generate_calm_data.py --n-prompts 300
"""
import argparse

from emotional_instability.training import generate_calm_conversations


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma-3-27b-it")
    ap.add_argument("--n-prompts", type=int, default=300)
    args = ap.parse_args()

    path = generate_calm_conversations(model_key=args.model, n_prompts=args.n_prompts)
    print(f"calm conversations -> {path}")


if __name__ == "__main__":
    main()
