#!/usr/bin/env python
"""Run the Petri-style open-ended emotion elicitation (Section 4.2, Figure 6).

  python scripts/run_petri.py --targets gemma-3-27b-it gemma-3-27b-dpo gemini-2.5-flash
  python scripts/run_petri.py --n-per-emotion 2 --max-turns 6   # smoke test
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from emo_instability.config import load_config
from emo_instability.petri import run_petri


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--targets", nargs="*", default=None)
    ap.add_argument("--n-per-emotion", type=int, default=10)
    ap.add_argument("--max-turns", type=int, default=20)
    args = ap.parse_args()

    cfg = load_config(args.config)
    summary = run_petri(cfg, targets=args.targets,
                        n_per_emotion=args.n_per_emotion, max_turns=args.max_turns)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
