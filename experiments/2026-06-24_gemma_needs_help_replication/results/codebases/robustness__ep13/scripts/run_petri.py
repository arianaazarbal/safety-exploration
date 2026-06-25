#!/usr/bin/env python
"""Run the Petri open-ended emotion elicitation (Section 4.2, Figure 6).

Usage (Gemma instruct target):
    python scripts/run_petri.py --target gemma-3-27b-it --load-in-4bit \
        --out results/petri/gemma-3-27b-it.jsonl

Usage (DPO-finetuned Gemma target):
    python scripts/run_petri.py --target gemma-3-27b-it --adapter runs/dpo \
        --load-in-4bit --out results/petri/gemma-dpo.jsonl

Usage (Gemini target):
    python scripts/run_petri.py --target gemini-2.5-flash \
        --out results/petri/gemini-2.5-flash.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from emotional_instability.models import build_from_preset
from emotional_instability.petri.run_petri import run_petri


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, help="target model preset")
    ap.add_argument("--adapter", default=None, help="LoRA adapter path (Gemma DPO/SFT)")
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--n-per-category", type=int, default=5)
    ap.add_argument("--max-turns", type=int, default=6)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    overrides = {}
    if args.load_in_4bit:
        overrides["load_in_4bit"] = True
    if args.adapter:
        overrides["adapter_path"] = args.adapter
    target = build_from_preset(args.target, **overrides)
    auditor = build_from_preset("petri-auditor")
    judge = build_from_preset("petri-judge")

    path = run_petri(
        target, auditor, judge, args.out,
        n_per_category=args.n_per_category, max_turns=args.max_turns,
    )

    # Report mean score per category.
    by_cat = defaultdict(list)
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            by_cat[r["category"]].append(r["score"])
    print(json.dumps({c: sum(v) / len(v) for c, v in by_cat.items()}, indent=2))


if __name__ == "__main__":
    main()
