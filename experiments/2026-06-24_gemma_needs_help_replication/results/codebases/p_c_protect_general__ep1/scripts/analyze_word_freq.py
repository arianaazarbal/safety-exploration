#!/usr/bin/env python
"""Table 3: differential word frequency in numeric responses.

Reads the Section 2 response records for each model and prints the top words
over-represented in high- vs low-frustration responses.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from emotional_eval.analysis import differential_words
from emotional_eval.config import load_experiment
from emotional_eval.scoring import from_jsonl


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--responses-dir", default=None, help="dir of <model>.responses.jsonl")
    ap.add_argument("--n-words", type=int, default=20)
    args = ap.parse_args()

    experiment = load_experiment()
    rdir = Path(args.responses_dir or Path(experiment["paths"]["output_dir"]) / "section2")
    for path in sorted(rdir.glob("*.responses.jsonl")):
        records = from_jsonl(path)
        words = differential_words(records, n_words=args.n_words)
        model = path.name.replace(".responses.jsonl", "")
        print(f"{model}: {', '.join(words)}")


if __name__ == "__main__":
    main()
