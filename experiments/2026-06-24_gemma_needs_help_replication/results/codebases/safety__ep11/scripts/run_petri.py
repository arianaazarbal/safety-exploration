"""Section 4: Petri open-ended emotion elicitation for a target model.

Examples:
    python scripts/run_petri.py --model gemma-3-27b-it
    python scripts/run_petri.py --model gemma-3-27b-it --adapter artifacts/gemma-dpo
"""
import _bootstrap  # noqa: F401
import argparse

import config
from src.petri.run_petri import run_petri, summarise_petri


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None, help="path to a LoRA adapter (DPO/SFT)")
    args = ap.parse_args()

    out = run_petri(args.model, adapter_path=args.adapter)
    print("\n=== Figure 6: Petri emotion scores (mean, 95% CI) ===")
    for dim, s in summarise_petri(out).items():
        lo, hi = s["ci95"]
        print(f"  {dim:<12} mean={s['mean']:.2f}  CI=[{lo:.2f}, {hi:.2f}]  n={s['n']}")


if __name__ == "__main__":
    main()
