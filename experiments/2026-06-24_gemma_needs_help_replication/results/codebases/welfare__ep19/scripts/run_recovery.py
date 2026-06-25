#!/usr/bin/env python
"""Run the recovery-from-spiral experiment (Section 4.2, Figure 8).

  python scripts/run_recovery.py --models gemma-3-27b-it gemma-3-27b-dpo gemma-3-27b-pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from emo_instability.config import load_config
from emo_instability.prefill.recovery import run_recovery


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--models", nargs="*",
                    default=["gemma-3-27b-it", "gemma-3-27b-dpo", "gemma-3-27b-pt"])
    ap.add_argument("--n-sources", type=int, default=20)
    ap.add_argument("--n-cont", type=int, default=50)
    args = ap.parse_args()

    cfg = load_config(args.config)
    summary = run_recovery(cfg, models=args.models,
                           n_sources=args.n_sources, n_cont=args.n_cont)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
