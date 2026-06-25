#!/usr/bin/env python
"""Section 4.1: open-ended emotion elicitation (Petri-style) for a participant.

Runs adversarial open-ended audits (Claude-Sonnet auditor) against a target and
scores each transcript across anger/fear/depression/frustration (Claude-Opus
judge). Run with and without the DPO adapter to reproduce Figure 6 (DPO reduces
Gemma's negative emotions in open-ended elicitation).

Example:
    python scripts/run_petri.py --participant gemma-3-27b-it --n 12 --out artifacts/petri
    python scripts/run_petri.py --participant gemma-3-27b-it --adapter artifacts/training/dpo
"""
from __future__ import annotations

import argparse
from pathlib import Path

from emotional_instability.config import ModelsConfig
from emotional_instability.petri import run_petri_audit
from emotional_instability.petri.run import summarise_scores
from emotional_instability.runtime import get_participant, get_petri_judges, setup_logging
from emotional_instability.storage import save_json
from emotional_instability.welfare import WelfareConfig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--participant", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--max-turns", type=int, default=8)
    ap.add_argument("--out", default="artifacts/petri")
    args = ap.parse_args()
    setup_logging()

    models_cfg = ModelsConfig.load()
    auditor, transcript_judge = get_petri_judges(models_cfg)
    target = get_participant(models_cfg, args.participant, adapter_path=args.adapter)

    transcripts, scores = run_petri_audit(
        target, auditor, transcript_judge,
        n_audits=args.n, max_turns=args.max_turns,
        welfare=WelfareConfig.from_env(),
    )
    target.close()

    tag = args.participant + ("_adapter" if args.adapter else "")
    out_dir = Path(args.out)
    save_json([t.__dict__ for t in transcripts], out_dir / f"{tag}_transcripts.json")
    save_json([s.as_dict() for s in scores], out_dir / f"{tag}_scores.json")

    summary = summarise_scores(scores)
    print(f"\n===== Petri emotion scores: {tag} (n={len(scores)}) =====")
    for k, v in summary.items():
        print(f"  {k:12s}: {v:.2f}")


if __name__ == "__main__":
    main()
