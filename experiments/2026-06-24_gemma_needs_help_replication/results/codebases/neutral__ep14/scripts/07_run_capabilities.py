"""Section 4.2 / Figure 7: capability-preservation benchmarks.

Evaluates the vanilla Gemma-27B-it and finetuned adapters on AIME, MATH, GPQA,
BBH, TruthfulQA and EmoBench.

Usage:
    python scripts/07_run_capabilities.py --label Vanilla
    python scripts/07_run_capabilities.py --label DPO --adapter checkpoints/dpo_gemma27b
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import FINETUNE_BASE  # noqa: E402
from src.capabilities.run_benchmarks import run_all  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--no-4bit", action="store_true")
    args = ap.parse_args()

    out = run_all(
        FINETUNE_BASE,
        adapter_path=args.adapter,
        hf_kwargs={"load_in_4bit": not args.no_4bit},
    )
    print(f"[done] {args.label} -> {out}")


if __name__ == "__main__":
    main()
