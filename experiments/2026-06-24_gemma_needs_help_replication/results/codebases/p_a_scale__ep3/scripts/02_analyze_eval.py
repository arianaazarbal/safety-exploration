#!/usr/bin/env python
"""Section 2 analysis: aggregate metrics (Figures 1-3), differential word lists
(Table 8), and render figures. Reads the scored rollouts produced by
01_run_eval.py.

Examples:
  python scripts/02_analyze_eval.py
  python scripts/02_analyze_eval.py --models gemma-3-27b-it gemini-2.5-flash
"""
from pathlib import Path

import pandas as pd

from _bootstrap import boot, common_parser

from eilm.analysis import metrics, plots
from eilm.analysis.word_freq import differential_words
from eilm.utils.io import write_json


def main():
    p = common_parser(__doc__)
    p.add_argument("--models", nargs="*", default=None)
    args = p.parse_args()
    cfg, registry, logger = boot(args)

    models = args.models or cfg["eval_targets"]
    # Include any finetuned store-names that exist (e.g. dpo) if present.
    tables = metrics.assemble_all(cfg, models)
    if not tables:
        logger.warning("No scores found for %s. Run 01_run_eval.py first.", models)
        return

    results_dir = cfg.path("results")
    tables["category_metrics"].to_csv(results_dir / "category_metrics.csv", index=False)
    tables["headline"].to_csv(results_dir / "headline.csv", index=False)
    tables["per_turn_extended"].to_csv(results_dir / "per_turn_extended.csv", index=False)
    tables["per_turn_wildchat"].to_csv(results_dir / "per_turn_wildchat.csv", index=False)
    logger.info("Headline (Figure 1):\n%s", tables["headline"].to_string(index=False))

    # Differential word lists (Table 8)
    word_tables = {}
    for m in models:
        rp = cfg.path("data") / "rollouts" / f"{m}.jsonl"
        sp = cfg.path("data") / "scores" / f"{m}.jsonl"
        if rp.exists() and sp.exists():
            word_tables[m] = differential_words(rp, sp)
    write_json(results_dir / "differential_words.json", word_tables)

    plots.render_all(tables, cfg.path("figures"))
    logger.info("Wrote results to %s and figures to %s", results_dir, cfg.path("figures"))


if __name__ == "__main__":
    main()
