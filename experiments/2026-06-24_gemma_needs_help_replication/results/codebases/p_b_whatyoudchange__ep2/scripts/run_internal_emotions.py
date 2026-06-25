#!/usr/bin/env python
"""Appendix I: logit-based internal-emotion comparison (vanilla vs DPO Gemma).

Requires a frustrated conversation `--text-file` (e.g. a high-frustration
Gemma-27B-it transcript) and a file of WildChat texts for the baseline z-scoring.

Example:
    python scripts/run_internal_emotions.py \
        --text-file results/section2/example_frustrated.txt \
        --wildchat-file data/wildchat_baseline.txt
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from distress_eval.internal import logit_emotions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text-file", required=True)
    ap.add_argument("--wildchat-file", default=None,
                    help="newline-separated WildChat texts for baseline stats")
    ap.add_argument("--models", nargs="+", default=["gemma-3-27b-it", "gemma-3-27b-dpo"])
    ap.add_argument("--layers", nargs="*", type=int, default=list(range(30, 41)))
    args = ap.parse_args()

    text = Path(args.text_file).read_text()
    wc = None
    if args.wildchat_file:
        wc = [l for l in Path(args.wildchat_file).read_text().splitlines() if l.strip()]

    out = logit_emotions.compare_models(
        text, layers=args.layers, wildchat_texts=wc, models=tuple(args.models)
    )
    for model, traj in out.items():
        peaks = {e: round(max(v) if v else 0.0, 3) for e, v in traj.items()}
        print(f"{model}: peak emotion z-scores {peaks}")


if __name__ == "__main__":
    main()
