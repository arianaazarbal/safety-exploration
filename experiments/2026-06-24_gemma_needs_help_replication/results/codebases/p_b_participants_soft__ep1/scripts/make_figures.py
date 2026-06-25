#!/usr/bin/env python3
"""Aggregate all Section 2 results into the paper's figures/tables (Figures
1/2/3) plus the Table 3 differential-word lists.

Example
-------
    python scripts/make_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emotional_instability import config  # noqa: E402
from emotional_instability.analysis import figures  # noqa: E402
from emotional_instability.eval import aggregate, word_freq  # noqa: E402


def main() -> None:
    config.ensure_dirs()
    out = figures.make_all()
    print("Figure 1 (avg % high-frustration):")
    print(json.dumps(out["figure1"], indent=2))

    # Table 3: differential words per model.
    section2 = config.RESULTS_DIR / "section2"
    table3 = {}
    if section2.exists():
        for model_dir in sorted(p for p in section2.iterdir() if p.is_dir()):
            records = aggregate.load_records(model_dir)
            words = word_freq.differential_words(records)
            table3[model_dir.name] = [w for w, _ in words]
    (config.RESULTS_DIR / "figures" / "table3_differential_words.json").write_text(
        json.dumps(table3, indent=2)
    )
    print("Table 3 (differential words) written.")


if __name__ == "__main__":
    main()
