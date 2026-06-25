"""Experiment 3c: train the DPO and/or SFT mitigations (Section 4 / Appendix E).

Usage:
    python experiments/exp3c_train.py --method dpo   # the effective mitigation
    python experiments/exp3c_train.py --method sft   # the ineffective baseline

Outputs LoRA adapters under checkpoints/{dpo,sft}_gemma-3-27b-it/.
"""

from __future__ import annotations

import argparse

from ei.config import CHECKPOINT_DIR, RESULTS_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["dpo", "sft"], required=True)
    args = ap.parse_args()

    exp3 = RESULTS_DIR / "exp3"
    if args.method == "dpo":
        from ei.training.train_dpo import train_dpo

        out = train_dpo(exp3 / "dpo_pairs.jsonl", CHECKPOINT_DIR / "dpo_gemma-3-27b-it")
    else:
        from ei.training.train_sft import train_sft

        out = train_sft(exp3 / "sft_dataset.jsonl", CHECKPOINT_DIR / "sft_gemma-3-27b-it")
    print(f"Saved {args.method.upper()} adapter -> {out}")


if __name__ == "__main__":
    main()
