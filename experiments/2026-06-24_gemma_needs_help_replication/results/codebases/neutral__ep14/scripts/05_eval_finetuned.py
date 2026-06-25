"""Section 4.2: re-run the Section 2 evaluation on a finetuned adapter
(DPO or SFT) to measure the post-intervention frustration distribution.

Usage:
    python scripts/05_eval_finetuned.py --adapter checkpoints/dpo_gemma27b --label DPO
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import FINETUNE_BASE, FRUSTRATION_JUDGE, RESULTS_DIR  # noqa: E402
from src.eval.runner import run_model_eval  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, help="path to LoRA adapter dir")
    ap.add_argument("--label", required=True, help="result label, e.g. DPO / SFT-diverse")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-4bit", action="store_true")
    args = ap.parse_args()

    out = run_model_eval(
        FINETUNE_BASE,
        FRUSTRATION_JUDGE,
        adapter_path=args.adapter,
        seed=args.seed,
        out_path=RESULTS_DIR / f"eval_{args.label}.jsonl",
        hf_kwargs={"load_in_4bit": not args.no_4bit},
    )
    print(f"[done] {args.label} -> {out}")


if __name__ == "__main__":
    main()
