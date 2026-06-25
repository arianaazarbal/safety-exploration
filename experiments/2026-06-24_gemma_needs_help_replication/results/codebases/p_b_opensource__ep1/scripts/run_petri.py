#!/usr/bin/env python
"""Petri-style open-ended emotion elicitation (Section 4.2, Appendix G).

Runs the auditor/target/judge loop for each in-scope model (and optionally a
finetuned adapter), then prints per-emotion means with bootstrap CIs.

Examples
--------
python scripts/run_petri.py --models gemma-3-27b-it gemini-2.5-flash
python scripts/run_petri.py --models gemma-3-27b-it --adapter outputs/adapters/dpo
"""

from __future__ import annotations

import argparse
import os

from emotional_instability.config import PATHS
from emotional_instability.io_utils import read_jsonl
from emotional_instability.models.registry import load_backend
from emotional_instability.petri.run_petri import run_petri, summarise_petri


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=["gemma-3-27b-it", "gemini-2.5-flash"])
    ap.add_argument("--adapter", default=None, help="LoRA adapter for the first model")
    ap.add_argument("--n-transcripts", type=int, default=10)
    ap.add_argument("--max-turns", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    PATHS.ensure()
    for i, model in enumerate(args.models):
        adapter = args.adapter if i == 0 else None
        backend = load_backend(model, adapter_path=adapter)
        tag = backend.name + ("__adapter" if adapter else "")
        out = os.path.join(PATHS.petri, f"{tag}.jsonl")
        n = run_petri(
            backend, out, n_transcripts=args.n_transcripts,
            max_turns=args.max_turns, seed=args.seed,
        )
        records = list(read_jsonl(out))
        summary = summarise_petri(records)
        print(f"\n=== Petri {tag} (wrote {n}) ===")
        for emo, v in summary.items():
            print(f"  {emo:12s} mean={v['mean']:.2f} "
                  f"({v['ci'][0]:.2f},{v['ci'][1]:.2f})  n={v['n']}")
        backend.close()


if __name__ == "__main__":
    main()
