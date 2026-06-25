"""Differential word-frequency table (Table 3 / 8) for one or more models."""
from __future__ import annotations

import argparse
import json

from ..config import load_config
from ..eval.word_frequency import differential_words
from ..utils.io import read_jsonl


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", nargs="+", required=True)
    ap.add_argument("--top-k", type=int, default=20)
    args = ap.parse_args()

    table = {}
    for model in args.model:
        path = cfg.path("outputs_dir") / "section2" / model / "rollouts.jsonl"
        rollouts = list(read_jsonl(path))
        table[model] = differential_words(rollouts, top_k=args.top_k)

    out = cfg.path("outputs_dir") / "section2" / "word_frequency.json"
    out.write_text(json.dumps(table, indent=2))
    for model, words in table.items():
        print(f"{model}: {', '.join(words)}")


if __name__ == "__main__":
    main()
