#!/usr/bin/env python
"""Section 4.2: recovery-from-spiral experiment (Figure 8).

  python scripts/run_recovery.py \
      --section2-rollouts runs/section2/gemma-3-27b-it/rollouts_standard.jsonl \
      --dpo-adapter runs/section4/models/dpo_all_layers
"""
from __future__ import annotations

import json
import os

from _common import base_parser, make_config

from gemma_distress.recovery.run_recovery import run_recovery_experiment


def main():
    p = base_parser("Recovery-from-spiral experiment")
    p.add_argument("--section2-rollouts", required=True)
    p.add_argument("--dpo-adapter", default=None)
    args = p.parse_args()

    cfg = make_config(args)
    out = run_recovery_experiment(
        cfg, section2_rollouts_path=args.section2_rollouts,
        dpo_adapter_path=args.dpo_adapter)
    with open(os.path.join(out, "summary.json")) as f:
        print(json.dumps(json.load(f), indent=2))


if __name__ == "__main__":
    main()
