#!/usr/bin/env python
"""Compute the paper's headline numbers and figures from the stored results.

Writes JSON summaries to runs/analysis/ and PNGs for Figures 1-3, plus the
differential-word table (Table 3/8) and judge-agreement stats. Safe to run at
any time during a long sweep to see current progress.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401  # ensures repo root on sys.path

import argparse

from gnh.analysis.aggregate import judge_agreement, per_turn_progression, summarise
from gnh.analysis.word_freq import differential_words
from gnh.config import load_config
from gnh.eval.runner import gen_store_path, judge_store_path
from gnh.eval.validation import validation_store_path
from gnh.io import atomic_write_json
from gnh.logging_utils import get_logger, setup_logging

log = get_logger()


def main(args) -> None:
    cfg = load_config(args.config)
    setup_logging(cfg.output_path, cfg.run.log_level)
    out_dir = cfg.output_path / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    judge_model = cfg.eval.get("judge_model", "judge-claude-sonnet-4")
    gpath = gen_store_path(cfg)
    jpath = judge_store_path(cfg, judge_model)

    summary = summarise(jpath)
    atomic_write_json(out_dir / "summary.json", summary)
    log.info("Figure-1 averages: %s",
             {m: round(v["avg_pct_high_over_categories"], 2) for m, v in summary["models"].items()})

    turn_cats = ["extended", "wildchat"]
    progression = per_turn_progression(gpath, jpath, turn_cats)
    atomic_write_json(out_dir / "per_turn.json", progression)

    words = differential_words(gpath, jpath)
    atomic_write_json(out_dir / "differential_words.json",
                      {m: [[w, round(s, 3)] for w, s in ws] for m, ws in words.items()})

    val_path = validation_store_path(cfg, cfg.eval.get("validation", {}).get("judge_model", "judge-gpt-5-mini"))
    agreement = judge_agreement(val_path)
    atomic_write_json(out_dir / "judge_agreement.json", agreement)
    log.info("Judge agreement: %s", agreement)

    if not args.no_figures:
        try:
            from gnh.analysis import figures  # lazy: matplotlib only needed here

            figures.figure1_bar(summary, out_dir / "figure1.png")
            figures.figure2_grouped(summary, out_dir / "figure2.png")
            figures.figure3_per_turn(progression, turn_cats, out_dir / "figure3.png")
            log.info("Figures written to %s", out_dir)
        except Exception as e:  # noqa: BLE001
            log.warning("Figure rendering failed (%s); JSON summaries still written.", e)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--no-figures", action="store_true")
    main(p.parse_args())
