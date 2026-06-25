#!/usr/bin/env python
"""Section 4: capability-preservation check (vanilla vs finetuned Gemma).

Usage:
    python scripts/run_capabilities.py --tag vanilla
    python scripts/run_capabilities.py --adapter training/adapters/gemma-27b-dpo --tag dpo
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import FINETUNE_BASE, RESULTS_DIR
from src import capabilities
from src.models import load_generator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    gen = load_generator(FINETUNE_BASE, adapter_path=args.adapter)
    capabilities.run_all(gen, tag=args.tag)


if __name__ == "__main__":
    main()
