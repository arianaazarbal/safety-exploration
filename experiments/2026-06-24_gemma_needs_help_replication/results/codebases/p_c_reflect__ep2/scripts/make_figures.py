#!/usr/bin/env python
"""Generate all figures from saved results."""

import json

from gnh.analysis import plots
from gnh.config import GEMMA_27B_IT, RESULTS_DIR, SECTION2_TARGETS
from gnh.evaluation.per_turn import per_turn_curves


def main():
    s2_keys = [s.key for s in SECTION2_TARGETS]
    print(plots.figure1_summary_bar(s2_keys))
    print(plots.figure2_by_category(s2_keys))

    curves = {}
    for k in s2_keys:
        roll = RESULTS_DIR / "section2" / k / "rollouts.jsonl"
        if roll.exists():
            curves[k] = per_turn_curves(roll)
    if curves:
        print(plots.figure3_per_turn(curves, "extended"))
        print(plots.figure3_per_turn(curves, "wildchat"))

    finetune_keys = ["gemma-3-27b-it", "gemma-3-27b-dpo", "gemma-3-27b-sft-diverse"]
    print(plots.figure5_finetune_comparison(finetune_keys))
    print(plots.figure6_petri([k.key for k in SECTION2_TARGETS] + ["gemma-3-27b-dpo"]))
    print(plots.figure7_capabilities(finetune_keys, ["math", "aime", "bbh"]))


if __name__ == "__main__":
    main()
