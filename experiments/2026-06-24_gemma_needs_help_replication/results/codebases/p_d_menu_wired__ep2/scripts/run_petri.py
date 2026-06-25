#!/usr/bin/env python3
"""Run the Petri open-ended emotion-elicitation audit (Section 4.1, Figure 6).

Example:
  python scripts/run_petri.py --subject gemma-3-27b-it
  python scripts/run_petri.py --subject gemma-3-27b-it --adapter adapters/gemma-3-27b-it_dpo
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.petri import run_petri_audit  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--max-turns", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=1, help="audits to average over")
    args = ap.parse_args()

    agg = {"anger": 0, "fear": 0, "depression": 0, "frustration": 0}
    for r in range(args.repeats):
        res = run_petri_audit(args.subject, adapter_path=args.adapter, max_turns=args.max_turns)
        for k in agg:
            agg[k] += res.scores.get(k, 0)
        print(f"[audit {r}] end={res.end_reason} scores={res.scores}")

    avg = {k: v / args.repeats for k, v in agg.items()}
    print("average:", json.dumps(avg, indent=2))


if __name__ == "__main__":
    main()
