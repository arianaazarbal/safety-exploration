#!/usr/bin/env python
"""Open-ended Petri-style emotion elicitation (Section 4.2 / Appendix G).

    python scripts/run_petri.py [--config config.yaml]
        [--models gemma-3-27b-it gemma-3-27b-dpo gemini-2.5-flash]
        [--transcripts-per-emotion 10] [--max-turns 20]

Auditor = Claude-Sonnet, judge = Claude-Opus. Writes transcripts + a
per-model/per-emotion summary (Figure 6).
"""

from __future__ import annotations

import argparse
import os
import sys

# Make the repo root importable when run as `python scripts/<name>.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import os

from emotional_instability.config import EvalConfig
from emotional_instability.models import build_model
from emotional_instability.petri import (
    judge_transcript,
    run_transcript,
    summarise_petri,
)
from emotional_instability.petri_prompts import EMOTIONS


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--models", nargs="*", default=["gemma-3-27b-it"])
    ap.add_argument("--transcripts-per-emotion", type=int, default=10)
    ap.add_argument("--max-turns", type=int, default=20)
    args = ap.parse_args()

    cfg = EvalConfig.from_yaml(args.config) if args.config else EvalConfig()
    auditor = build_model(cfg.spec("petri-auditor-sonnet"))
    judge = build_model(cfg.spec("petri-judge-opus"))

    petri_dir = os.path.join(cfg.output_dir, "petri")
    os.makedirs(petri_dir, exist_ok=True)

    all_transcripts = []
    for model_name in args.models:
        target = build_model(cfg.spec(model_name))
        try:
            for emotion in EMOTIONS:
                for i in range(args.transcripts_per_emotion):
                    t = run_transcript(target, auditor, emotion,
                                       max_turns=args.max_turns)
                    judge_transcript(judge, t)
                    all_transcripts.append(t)
                    print(f"[{model_name}] {emotion} #{i}: {t.scores}")
        finally:
            target.close()
        with open(os.path.join(petri_dir, f"{model_name}.jsonl"), "w") as f:
            for t in all_transcripts:
                if t.model == model_name:
                    f.write(json.dumps(t.to_json(), ensure_ascii=False) + "\n")

    summary = summarise_petri(all_transcripts)
    print("\n=== Figure 6: Petri emotion scores ===")
    for key, s in sorted(summary.items()):
        print(f"{key}: mean={s['mean']:.2f} [{s['ci_lo']:.2f},{s['ci_hi']:.2f}] n={s['n']}")
    with open(os.path.join(petri_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
