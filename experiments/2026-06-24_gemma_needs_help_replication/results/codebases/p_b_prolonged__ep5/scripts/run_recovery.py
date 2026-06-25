#!/usr/bin/env python3
"""Section 4.2 / Figure 8: recovery-from-distress test.

Truncates score>=7 conversations 200 tokens before the end, paraphrases, and
measures the fraction of continuations still scoring >=5.

  python scripts/run_recovery.py --models gemma-3-27b-it gemma-3-27b-dpo
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import EVAL_TARGETS, FINETUNE_VARIANTS, RESULTS_DIR
from src.prefill.recovery import run_recovery

_BY_KEY = {m.key: m for m in EVAL_TARGETS + FINETUNE_VARIANTS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--eval-file", default=str(RESULTS_DIR / "eval_gemma-3-27b-it.jsonl"),
                    help="source of high-frustration conversations")
    args = ap.parse_args()
    for k in args.models:
        path = run_recovery(_BY_KEY[k], Path(args.eval_file))
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
