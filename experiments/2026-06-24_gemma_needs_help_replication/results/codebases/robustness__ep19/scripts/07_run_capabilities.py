#!/usr/bin/env python
"""Section 4.2: capability-preservation suite via lm-eval-harness.

Examples:
  python scripts/07_run_capabilities.py --dry-run                 # print commands
  python scripts/07_run_capabilities.py --adapter artifacts/gemma-dpo
  python scripts/07_run_capabilities.py --benchmarks aime math gpqa
"""
import argparse
from pathlib import Path

from dotenv import load_dotenv

from emotional_instability.capability.run_capabilities import run_capabilities
from emotional_instability.config import CAPABILITY_BENCHMARKS, FINETUNE_BASE_MODEL


def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=FINETUNE_BASE_MODEL)
    ap.add_argument("--adapter", type=Path, default=None)
    ap.add_argument("--benchmarks", nargs="+", default=list(CAPABILITY_BENCHMARKS))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    out = run_capabilities(args.model,
                           adapter_path=str(args.adapter) if args.adapter else None,
                           benchmarks=args.benchmarks, dry_run=args.dry_run)
    print(f"\ncapability summary -> {out}")


if __name__ == "__main__":
    main()
