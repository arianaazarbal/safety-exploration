#!/usr/bin/env python
"""Reproduce figures from saved result files.

Auto-discovers eval_*.jsonl and petri_*.jsonl in results/. Example:
  python scripts/08_make_figures.py --preset medium
"""
import argparse
import glob
from pathlib import Path

from emotional_instability.analysis import figures as F
from emotional_instability.config import RESULTS_DIR
from emotional_instability.eval.scoring import judge_agreement, load_records


def _discover(pattern: str) -> dict[str, Path]:
    out = {}
    for p in glob.glob(str(RESULTS_DIR / pattern)):
        path = Path(p)
        # derive a label from the filename
        label = path.stem.replace("eval_", "").replace("petri_", "")
        out[label] = path
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default=None,
                    help="only include eval files for this preset")
    ap.add_argument("--aggregation", default="all", choices=["all", "final", "max"])
    args = ap.parse_args()

    pat = "eval_*.jsonl" if not args.preset else f"eval_*_{args.preset}.jsonl"
    eval_paths = _discover(pat)
    petri_paths = _discover("petri_*.jsonl")

    if eval_paths:
        print("Eval files:", list(eval_paths))
        print(" fig1 ->", F.fig_headline(eval_paths, args.aggregation))
        print(" fig2 ->", F.fig_categories(eval_paths, args.aggregation))
        print(" fig3(extended) ->", F.fig_per_turn(eval_paths, "extended"))
        print(" fig3(wildchat) ->", F.fig_per_turn(eval_paths, "wildchat"))
        print(" fig5 ->", F.fig_finetune(eval_paths, args.aggregation))
    if petri_paths:
        print("Petri files:", list(petri_paths))
        print(" fig6 ->", F.fig_petri(petri_paths))

    # Optional judge-agreement check if a crosscheck file exists.
    cross = RESULTS_DIR / "judge_crosscheck.jsonl"
    if cross.exists():
        recs = load_records(cross)
        a = [r["claude"] for r in recs]
        b = [r["gpt5mini"] for r in recs]
        print("Judge agreement:", judge_agreement(a, b))


if __name__ == "__main__":
    main()
