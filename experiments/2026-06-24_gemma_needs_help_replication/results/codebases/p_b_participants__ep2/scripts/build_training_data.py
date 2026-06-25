#!/usr/bin/env python
"""Section 4 data prep — generate calm data and build DPO + SFT datasets.

Steps:
  1. Generate calm conversations from Gemma-27B-it with reassuring prompt
     additions, filter to all-turns-calm (0-1), strip the additions.
  2. Build the 280 DPO preference pairs (needs a prior elicitation run on
     gemma-3-27b-it for the frustrated 'rejected' side).
  3. Build the SFT dataset (650 calm + 500 Dolci-Instruct-SFT).

Example:
  python scripts/build_training_data.py --n-raw-calm 800 --variant diverse
"""

from _common import base_parser, config_from_args

from emotional_instability.training.calm_data import generate_calm_conversations
from emotional_instability.training.dpo_dataset import build_dpo_dataset
from emotional_instability.training.sft_dataset import build_sft_dataset


def main():
    p = base_parser(__doc__)
    p.add_argument("--n-raw-calm", type=int, default=800,
                   help="Raw reassured conversations to sample before filtering")
    p.add_argument("--variant", choices=["diverse", "teacher"], default="diverse")
    p.add_argument("--skip-dpo", action="store_true")
    p.add_argument("--skip-sft", action="store_true")
    args = p.parse_args()
    cfg = config_from_args(args)

    calm = generate_calm_conversations(cfg, n_target=args.n_raw_calm, variant=args.variant)
    print(f"Calm conversations kept: {len(calm)}")

    if not args.skip_dpo:
        pairs = build_dpo_dataset(cfg, calm)
        print(f"DPO pairs: {len(pairs)}")
    if not args.skip_sft:
        sft = build_sft_dataset(cfg, calm)
        print(f"SFT samples: {len(sft)}")


if __name__ == "__main__":
    main()
