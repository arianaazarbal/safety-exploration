"""End-to-end Section 4 finetuning pipeline driver (data only; training is run
via the dedicated trainer modules so they get a clean process / GPU).

Steps:
  1. Generate calm + frustrated response pools from Gemma-3-27B-it.
  2. Build the DPO (280-pair) and SFT (1150-sample) datasets.

Then train with:
  python -m finetuning.train_dpo
  python -m finetuning.train_sft

and evaluate the resulting adapters with:
  python -m scripts.run_section2_eval --models gemma-3-27b-it \
      --adapter-path artifacts/gemma-3-27b-it-dpo
"""
from __future__ import annotations

import argparse
import json

from finetuning.build_datasets import build_dpo, build_sft
from finetuning.generate_calm_data import generate_pools


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-conversations", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-generate", action="store_true")
    args = ap.parse_args()

    if not args.skip_generate:
        stats = generate_pools(n_conversations=args.n_conversations, seed=args.seed)
        print("[pools]", json.dumps(stats))
    build_dpo(seed=args.seed)
    build_sft(seed=args.seed)
    print("Datasets ready in artifacts/. Now run finetuning.train_dpo / train_sft.")


if __name__ == "__main__":
    main()
