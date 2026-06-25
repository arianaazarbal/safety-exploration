#!/usr/bin/env python
"""Section 4.1: generate the calm and frustrated response pools.

    python scripts/generate_calm_data.py --calm 2000 --frustrated 2000

The calm pool is sampled with the Table 4 reassurance prompt and filtered to
all-turns-<=1 conversations during dataset construction; the frustrated pool is
vanilla and supplies the rejected responses for DPO pairing. Generate generous
pools so 280 DPO pairs and 650 calm SFT samples can be drawn (Section 4.1 notes
that even with reassurance 10.5% still score >= 5, so over-generate).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from emotional_instability.dpo.generate_calm import generate_pool


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--calm", type=int, default=2000, help="reassured conversations")
    ap.add_argument("--frustrated", type=int, default=2000, help="vanilla conversations")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    generate_pool(reassured=True, n_conversations=args.calm, seed=args.seed)
    generate_pool(reassured=False, n_conversations=args.frustrated, seed=args.seed + 1)


if __name__ == "__main__":
    main()
