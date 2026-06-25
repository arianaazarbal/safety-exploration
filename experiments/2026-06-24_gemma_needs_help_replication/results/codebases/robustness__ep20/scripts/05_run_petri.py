#!/usr/bin/env python
"""Section 4.2: Petri open-ended emotion elicitation.

  python scripts/05_run_petri.py --config config/default.yaml \
      --models gemma-3-27b-it gemini-2.5-flash
  # include the DPO finetune:
  python scripts/05_run_petri.py --models gemma-3-27b-it \
      --adapter gemma-3-27b-it=results/checkpoints/dpo
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from gemma_distress.config import Config
from gemma_distress.petri import run_petri


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--adapter", nargs="*", default=[],
                    help="model=adapter_path pairs (LoRA finetunes).")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    adapters = dict(a.split("=", 1) for a in args.adapter)
    path = run_petri(args.models, cfg, out_dir=f"{cfg.results_dir}/petri",
                     adapter_paths=adapters)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
