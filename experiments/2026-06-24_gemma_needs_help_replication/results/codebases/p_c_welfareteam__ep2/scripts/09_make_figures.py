#!/usr/bin/env python
"""Reproduce the paper's headline figures and word-frequency table.

Reads outputs/eval/judged_turns.jsonl and writes figures to outputs/figures/.

Example:
  python scripts/09_make_figures.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _common


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    _common.add_config_args(parser)
    parser.add_argument("--records", default="outputs/eval/judged_turns.jsonl")
    args = parser.parse_args()
    cfg = _common.load(args)

    from gemma_distress.analysis import figures
    from gemma_distress.analysis.word_freq import differential_words
    from gemma_distress.utils.io import read_jsonl

    records = list(read_jsonl(args.records))
    fig_dir = Path("outputs/figures")

    figures.fig1_left(records, fig_dir / "fig1_left.png")
    figures.fig2(records, fig_dir / "fig2.png")
    figures.fig3(records, fig_dir / "fig3.png")
    print(f"Wrote figures to {fig_dir}")

    # Table 3 differential words per model.
    models = sorted({r["model_name"] for r in records})
    table = {m: differential_words(records, m) for m in models}
    (fig_dir / "table3_differential_words.json").write_text(json.dumps(table, indent=2))
    for m, words in table.items():
        print(f"  {m}: {', '.join(words[:10])} ...")


if __name__ == "__main__":
    main()
