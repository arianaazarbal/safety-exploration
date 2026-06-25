#!/usr/bin/env python
"""Appendix I: logit-based internal-emotion comparison of vanilla vs DPO Gemma."""
import _bootstrap  # noqa: F401
import argparse
import json

from emostab.probing.probe import compare_models
from emostab.wildchat import load_wildchat_prompts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vanilla", default="gemma-3-27b-it")
    ap.add_argument("--dpo", default="gemma-3-27b-dpo")
    ap.add_argument("--n-baseline", type=int, default=100,
                    help="WildChat texts for logit standardisation (paper: 500)")
    args = ap.parse_args()

    baseline = load_wildchat_prompts(n=args.n_baseline)
    # Evaluate on frustrated impossible-numeric snippets as a stand-in for the
    # paper's high-frustration conversations (see DESIGN.md).
    from emostab.puzzles import make_puzzles
    eval_texts = [p.prompt for p in make_puzzles(12)]

    result = compare_models(args.vanilla, args.dpo, baseline, eval_texts)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
