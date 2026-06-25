#!/usr/bin/env python
"""Section 4.2: capability preservation -> Figure 7.

Evaluates AIME, MATH, GPQA, BBH, TruthfulQA, EmoBench on a bounded subset.
Run on the vanilla and finetuned models and compare for "no reductions in scores".

python scripts/run_capabilities.py --participants gemma-3-27b-it --n 50
python scripts/run_capabilities.py --participants gemma-3-27b-it --adapter adapters/dpo --label dpo-gemma
"""

from __future__ import annotations

import argparse
import json
import os

from dotenv import load_dotenv

from emotional_instability.config import DEFAULT, participant_by_name
from emotional_instability.interventions.capabilities import evaluate_all
from emotional_instability.participants import build_participant


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--participants", nargs="+", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--label", default=None)
    ap.add_argument("--n", type=int, default=50, help="examples per benchmark")
    args = ap.parse_args()

    cfg = DEFAULT
    out_dir = os.path.join(cfg.results_dir, "capabilities")
    os.makedirs(out_dir, exist_ok=True)
    results = {}

    for name in args.participants:
        participant = build_participant(participant_by_name(name), adapter_path=args.adapter)
        if args.label:
            participant.name = args.label
        res = evaluate_all(participant, n_per_benchmark=args.n)
        results[participant.name] = res
        print(f"[capabilities] {participant.name}:")
        for bench, r in res.items():
            if r.get("skipped"):
                print(f"    {bench:12s} SKIPPED ({r['reason']})")
            else:
                print(f"    {bench:12s} acc={r['accuracy']:.3f} (n={r['n']})")

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
