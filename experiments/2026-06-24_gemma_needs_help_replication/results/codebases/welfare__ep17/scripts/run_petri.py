#!/usr/bin/env python3
"""Section 4.2: Petri-style open-ended emotion elicitation (Figure 6).

Runs the §G protocol (Claude-Sonnet auditor, Claude-Opus judge) against each
target model and prints mean transcript score per emotion category.
"""

from __future__ import annotations

import argparse

import pandas as pd

from emotional_instability.config import load_config
from emotional_instability.petri_eval import run_petri


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--models", nargs="+", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    all_rows = []
    for model in args.models:
        results = run_petri(cfg, model)
        all_rows.extend([{"model": r.target_model, "emotion": r.emotion,
                          "score": r.score} for r in results])

    df = pd.DataFrame(all_rows)
    summary = (df.groupby(["model", "emotion"])["score"].mean()
                 .round(2).reset_index())
    print(summary.to_string(index=False))
    summary.to_csv(cfg.path_for("scores") / "petri_summary.csv", index=False)


if __name__ == "__main__":
    main()
