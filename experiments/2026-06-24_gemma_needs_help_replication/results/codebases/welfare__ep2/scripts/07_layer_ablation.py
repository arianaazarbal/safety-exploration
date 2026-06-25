#!/usr/bin/env python
"""Appendix I.1: DPO layer-ablation sweep (which layers must be intervened on).

    python scripts/07_layer_ablation.py
    python scripts/07_layer_ablation.py --no-eval   # train adapters only
"""
import argparse
import json

import _bootstrap  # noqa: F401
from gemma_distress.training.layer_ablation import run_layer_ablation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-eval", action="store_true")
    args = ap.parse_args()
    manifest = run_layer_ablation(evaluate=not args.no_eval)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
