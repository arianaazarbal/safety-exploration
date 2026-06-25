#!/usr/bin/env python
"""Section 4 / Appendix F: SFT finetune Gemma-3-27B-it (diverse or teacher variant).

For the 'teacher' variant, first regenerate calm data with --variant teacher
(script 04) and rebuild SFT samples from it.

Usage:
    python scripts/07_train_sft.py --variant diverse
"""
from pathlib import Path

from _common import base_parser, cfg_from_args

from emotional_instability.training.sft import train_sft


def main():
    p = base_parser(__doc__)
    p.add_argument("--variant", choices=["diverse", "teacher"], default="diverse")
    p.add_argument("--samples", default=None, help="SFT samples jsonl (default: runs/training/sft_samples.jsonl)")
    args = p.parse_args()
    cfg = cfg_from_args(args)
    cfg["sft"]["variant"] = args.variant
    samples = args.samples or str(Path(cfg["run"]["output_dir"]) / "training" / "sft_samples.jsonl")
    out = train_sft(cfg, samples)
    print(f"SFT ({args.variant}) adapter saved to {out}")


if __name__ == "__main__":
    main()
