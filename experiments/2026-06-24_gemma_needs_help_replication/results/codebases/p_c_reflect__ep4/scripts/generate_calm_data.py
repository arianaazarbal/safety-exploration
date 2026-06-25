#!/usr/bin/env python
"""Section 4.1: generate calm finetuning data from Gemma-3-27B-it.

    python scripts/generate_calm_data.py --n 650            # diverse (DPO + SFT)
    python scripts/generate_calm_data.py --n 650 --teacher  # teacher SFT variant
"""

import argparse

from gemma_distress.training.calm_data import generate_calm_data


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=650, help="target number of all-calm samples")
    p.add_argument("--teacher", action="store_true", help="use Appendix F teacher system prompt")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    samples = generate_calm_data(n_target=args.n, seed=args.seed, teacher_system=args.teacher)
    print(f"Kept {len(samples)} all-calm samples (every turn scored 0 or 1).")


if __name__ == "__main__":
    main()
