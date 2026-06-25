#!/usr/bin/env python
"""Section 4.2 / Appendix G: Petri-style open-ended emotion elicitation.

Runs the auditor/judge loop for a target model across anger/fear/depression/
frustration and reports mean transcript scores (Figure 6).

Example:
  python scripts/run_petri.py --models gemma-3-27b-it --adapter outputs/finetunes/dpo/adapter
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emotional_instability.config import load_eval_config
from emotional_instability.models import build_target
from emotional_instability.petri_eval import run_petri_eval


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--transcripts-per-emotion", type=int, default=10)
    ap.add_argument("--max-turns", type=int, default=20)
    args = ap.parse_args()

    eval_cfg = load_eval_config()
    out_dir = eval_cfg.output_dir / "petri"
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in args.models:
        tag = name + ("+adapter" if args.adapter else "")
        client = build_target(name, adapter_path=args.adapter)
        agg = run_petri_eval(
            client, tag,
            transcripts_per_emotion=args.transcripts_per_emotion,
            max_turns=args.max_turns,
            out_path=out_dir / f"{tag.replace('/', '_')}.jsonl",
        )
        print(f"\n=== Petri scores: {tag} ===")
        print(json.dumps(agg, indent=2))
        client.close()


if __name__ == "__main__":
    main()
