#!/usr/bin/env python
"""Section 4 — open-ended (Petri-style) emotion elicitation.

Auditor = Claude-Sonnet-4, Judge = Claude-Opus-4 (paper defaults). Targets are
typically vanilla Gemma, the DPO fine-tune, and the SFT fine-tune.

Example
-------
python scripts/run_petri.py --targets gemma-3-27b-it gemma-3-27b-it-dpo \
    --transcripts-per-emotion 10
"""

from __future__ import annotations

import argparse

from emotional_instability.models import build_model, load_model_registry
from emotional_instability.petri import run_petri


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--targets", nargs="+", required=True)
    p.add_argument("--auditor", default="petri-auditor-claude-sonnet-4")
    p.add_argument("--judge", default="petri-judge-claude-opus-4")
    p.add_argument("--transcripts-per-emotion", type=int, default=10)
    p.add_argument("--max-turns", type=int, default=20)
    p.add_argument("--out-dir", default="outputs/petri")
    return p.parse_args()


def main():
    args = parse_args()
    registry = load_model_registry()
    auditor = build_model(args.auditor, registry)
    judge = build_model(args.judge, registry)

    for key in args.targets:
        print(f"=== Petri elicitation: {key} ===")
        target = build_model(key, registry)
        run_petri(
            target, auditor, judge,
            transcripts_per_emotion=args.transcripts_per_emotion,
            max_turns=args.max_turns,
            out_path=f"{args.out_dir}/{key}.jsonl",
        )


if __name__ == "__main__":
    main()
