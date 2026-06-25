#!/usr/bin/env python
"""Run the Section 3 base-vs-instruct prefilling experiment (Gemma).

  python scripts/run_prefill.py --models gemma-3-27b-pt gemma-3-27b-it
  python scripts/run_prefill.py --n-each 2 --n-cont 5   # smoke test
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from emo_instability.config import load_config
from emo_instability.prefill import run_prefill_experiment


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--models", nargs="*", default=None,
                    help="model names (default: gemma-3-27b-pt gemma-3-27b-it)")
    ap.add_argument("--n-each", type=int, default=10,
                    help="high-frustration source convos per domain (paper: 10)")
    ap.add_argument("--n-cont", type=int, default=50,
                    help="continuations per prefill (paper: 50)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    summary = run_prefill_experiment(
        cfg, models=args.models, n_each=args.n_each, n_cont=args.n_cont
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
