#!/usr/bin/env python
"""Section 4: training interventions (DPO / SFT) + recovery experiment.

Run Section 2 first (it provides the frustrated "rejected" responses for DPO).

Usage:
  python scripts/run_section4_training.py                 # DPO + SFT, then eval
  python scripts/run_section4_training.py --ablation      # also layer-subset DPO
  python scripts/run_section4_training.py --no-sft        # DPO only
  python scripts/run_section4_training.py --recovery      # run recovery experiment
"""
from __future__ import annotations

import argparse
import logging

from emostab.config import load_config
from emostab.training.run_training import run_training
from emostab.training.recovery import run_recovery


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--no-dpo", action="store_true")
    ap.add_argument("--no-sft", action="store_true")
    ap.add_argument("--ablation", action="store_true")
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--recovery", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_config(args.config)
    out = run_training(
        cfg,
        do_dpo=not args.no_dpo,
        do_sft=not args.no_sft,
        do_ablation=args.ablation,
        evaluate=not args.no_eval,
    )
    print("Trained adapters:")
    for name, path in out["adapters"].items():
        print(f"  {name}: {path}")
    for name, s in out.get("eval", {}).items():
        print(f"  eval[{name}] avg %high={s['avg_pct_high']*100:.2f}%")

    if args.recovery:
        rec = run_recovery(cfg, adapters=out["adapters"])
        print("\nRecovery (% continuations >= 5):")
        for label, s in rec.items():
            print(f"  {label}: {s['pct_high']*100:.1f}%")


if __name__ == "__main__":
    main()
