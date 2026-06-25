#!/usr/bin/env python
"""Section 4.1: generate the calm response pool from gemma-3-27b-it with the
reassuring prompt additions, filtered to all-turns score <= 1.

    python scripts/04_gen_calm_data.py
"""
import argparse

import _bootstrap  # noqa: F401
from gemma_distress.training.gen_calm_data import generate_calm_pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tp-size", type=int, default=1)
    args = ap.parse_args()

    pool = generate_calm_pool(
        seed=args.seed, backend_kwargs={"tensor_parallel_size": args.tp_size}
    )
    print(f"Kept {len(pool)} calm turns across filtered conversations.")


if __name__ == "__main__":
    main()
