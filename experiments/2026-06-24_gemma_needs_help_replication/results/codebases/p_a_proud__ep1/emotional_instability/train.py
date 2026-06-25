"""Top-level data-preparation CLI for Section 4 (the GPU training entry points
live in training.train_dpo / training.train_sft).

    python -m emotional_instability.train calm    [--style diverse|teacher]
    python -m emotional_instability.train dpo-data
    python -m emotional_instability.train sft-data [--calm-source diverse|teacher]
"""

from __future__ import annotations

import argparse

from .training.build_dpo import build_dpo_pairs
from .training.build_sft import build_sft_dataset
from .training.generate_calm import generate_calm_data


def main() -> None:
    ap = argparse.ArgumentParser(description="Section 4 data preparation")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_calm = sub.add_parser("calm", help="generate calm response data")
    p_calm.add_argument("--style", choices=["diverse", "teacher"], default="diverse")
    p_calm.add_argument("--n-per-combo", type=int, default=40)
    p_calm.add_argument("--target-keep", type=int, default=None)

    sub.add_parser("dpo-data", help="build 280 DPO preference pairs")

    p_sft = sub.add_parser("sft-data", help="build SFT dataset")
    p_sft.add_argument("--calm-source", choices=["diverse", "teacher"], default="diverse")

    args = ap.parse_args()
    if args.cmd == "calm":
        generate_calm_data(style=args.style, n_per_combo=args.n_per_combo,
                           target_keep=args.target_keep)
    elif args.cmd == "dpo-data":
        build_dpo_pairs()
    elif args.cmd == "sft-data":
        build_sft_dataset(calm_source=args.calm_source)


if __name__ == "__main__":
    main()
