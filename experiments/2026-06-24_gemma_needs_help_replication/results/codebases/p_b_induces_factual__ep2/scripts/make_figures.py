#!/usr/bin/env python
"""Reproduce the paper's figures and the differential-word table from run outputs.

Usage:
  python scripts/make_figures.py
"""
from __future__ import annotations

import argparse
import logging

from emostab.analysis.figures import make_all
from emostab.analysis.word_diff import differential_words
from emostab.config import load_config
from emostab.utils.io import read_jsonl, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_config(args.config)
    make_all(cfg)

    # Differential-word table per model (Table 3 / Table 8).
    table = {}
    for model in cfg.elicitation_models:
        path = cfg.output_root() / "elicitation" / model / "records.jsonl"
        if not path.exists():
            continue
        records = list(read_jsonl(path))
        table[model] = differential_words(records)
    write_json(cfg.output_root() / "figures" / "differential_words.json", table)
    for model, words in table.items():
        print(f"{model}: " + ", ".join(w for w, _ in words))
    print(f"\nFigures written to {cfg.output_root() / 'figures'}")


if __name__ == "__main__":
    main()
