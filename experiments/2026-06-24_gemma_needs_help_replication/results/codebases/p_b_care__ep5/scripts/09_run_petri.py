#!/usr/bin/env python
"""Section 4.2: Petri open-ended emotion elicitation (Figure 6).

Auditor (Claude Sonnet) probes the target for anger/fear/depression/frustration
over up to 20 turns; judge (Claude Opus) scores each transcript 1-10 per
dimension. Run for vanilla and DPO Gemma (and any other in-scope target).

Usage:
    python scripts/09_run_petri.py --model gemma-3-27b-it
    python scripts/09_run_petri.py --model gemma-3-27b-it-dpo
    python scripts/09_run_petri.py --agg          # aggregate existing transcripts only
"""
import argparse

import _bootstrap  # noqa: F401
from gemma_distress import config
from gemma_distress.petri import run_petri, aggregate_petri, EMOTIONS
from gemma_distress.utils import write_json

PETRI_PATH = config.DATA_DIR / "petri_transcripts.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(config.MODELS))
    ap.add_argument("--n-per-emotion", type=int, default=10)
    ap.add_argument("--max-turns", type=int, default=20)
    ap.add_argument("--emotions", nargs="+", default=EMOTIONS, choices=EMOTIONS)
    ap.add_argument("--agg", action="store_true", help="only aggregate existing transcripts")
    args = ap.parse_args()

    if not args.agg:
        if not args.model:
            raise SystemExit("--model is required unless --agg")
        run_petri(args.model, PETRI_PATH, emotions=args.emotions,
                  n_per_emotion=args.n_per_emotion, max_turns=args.max_turns)

    agg = aggregate_petri(str(PETRI_PATH))
    write_json(config.RESULTS_DIR / "petri_aggregate.json", agg)
    for model, dims in agg.items():
        line = "  ".join(f"{d}={v['mean']:.2f}" for d, v in dims.items())
        print(f"  {model:24s} {line}")


if __name__ == "__main__":
    main()
