#!/usr/bin/env python
"""Table 3 / Table 8: top-20 differential words (high vs low frustration) on
numeric responses, per model.

Reads the saved impossible-numeric scored rollouts and computes enrichment.

Example:
  python scripts/run_word_freq.py --model gemma-3-27b-it
"""
import _bootstrap  # noqa: F401

import argparse
import os

import config
from emotional_instability import io_utils
from emotional_instability.analysis import differential_words


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()

    in_dir = os.path.join(config.RESULTS_DIR, "section2", args.model)
    path = os.path.join(in_dir, "impossible_numeric_scores.jsonl")
    if not os.path.exists(path):
        raise SystemExit(f"Missing {path}; run run_section2.py for impossible_numeric first.")

    texts, scores = [], []
    for row in io_utils.read_jsonl(path):
        for turn in row.get("turns", []):
            if turn.get("score") is not None:
                texts.append(turn["assistant_response"])
                scores.append(turn["score"])

    result = differential_words(args.model, texts, scores)
    io_utils.write_json(os.path.join(in_dir, "differential_words.json"),
                        {"top_words": result.top_words, "enrichment": result.enrichment,
                         "n_high": result.n_high, "n_low": result.n_low})
    print(f"[{args.model}] top differential words:")
    print(", ".join(result.top_words))


if __name__ == "__main__":
    main()
