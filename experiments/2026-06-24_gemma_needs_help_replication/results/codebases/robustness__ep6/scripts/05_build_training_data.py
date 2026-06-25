#!/usr/bin/env python
"""Section 4.1: build the DPO preference pairs and the SFT dataset from the
generated calm data + the Gemma-27B-it frustrated responses."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from distress_eval.training.build_datasets import build_dpo, build_sft  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["dpo", "sft", "both"], default="both")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.which in ("dpo", "both"):
        build_dpo(seed=args.seed)
    if args.which in ("sft", "both"):
        build_sft(seed=args.seed)


if __name__ == "__main__":
    main()
