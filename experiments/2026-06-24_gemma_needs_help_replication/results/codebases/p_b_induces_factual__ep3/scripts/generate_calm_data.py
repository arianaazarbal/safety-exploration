#!/usr/bin/env python3
"""Generate calm + matched-frustrated training samples (Section 4.1).

Example:
    python scripts/generate_calm_data.py --n-puzzles 400 --style diverse
    python scripts/generate_calm_data.py --style teacher   # Appendix F variant
"""

import argparse

import _bootstrap  # noqa: F401
from emotional_instability.config import load_config
from emotional_instability.training.generate_calm import generate_training_samples


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--n-puzzles", type=int, default=400)
    ap.add_argument("--style", choices=["diverse", "teacher"], default="diverse")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    path = generate_training_samples(
        cfg, n_puzzles=args.n_puzzles, style=args.style, out_path=args.out
    )
    print(f"[done] calm/plain samples: {path}")


if __name__ == "__main__":
    main()
