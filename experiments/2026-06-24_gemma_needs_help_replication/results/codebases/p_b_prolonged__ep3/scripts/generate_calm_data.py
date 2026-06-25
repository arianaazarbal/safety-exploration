#!/usr/bin/env python
"""Generate calm response data from Gemma-3-27B-it (Section 4.1, Table 4).

Produces enough fully-calm conversations to cover the SFT (650) and DPO (chosen)
needs. Run both variants to reproduce the Appendix F SFT failure analysis.

Examples:
    python scripts/generate_calm_data.py --n 700 --variant diverse
    python scripts/generate_calm_data.py --n 700 --variant teacher
"""
from __future__ import annotations

import argparse

from gemma_distress import config
from gemma_distress.training.calm_data import generate_calm_conversations


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=config.SFT.n_calm + 100)
    p.add_argument("--variant", default=config.SFT_DIVERSE_VARIANT, choices=["diverse", "teacher"])
    p.add_argument("--max-turns", type=int, default=3)
    args = p.parse_args()
    path = generate_calm_conversations(args.n, variant=args.variant, max_turns=args.max_turns)
    print(f"calm data ({args.variant}): {path}")


if __name__ == "__main__":
    main()
