"""Section 4.2: capability-preservation benchmarks (Figure 7).

Run on vanilla and finetuned Gemma and compare.

Examples:
    python scripts/run_capabilities.py --model gemma-3-27b-it
    python scripts/run_capabilities.py --model gemma-3-27b-it --adapter artifacts/gemma-dpo
"""
import _bootstrap  # noqa: F401
import argparse

from src.capabilities.run_benchmarks import run_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    args = ap.parse_args()
    out = run_all(args.model, adapter_path=args.adapter)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
