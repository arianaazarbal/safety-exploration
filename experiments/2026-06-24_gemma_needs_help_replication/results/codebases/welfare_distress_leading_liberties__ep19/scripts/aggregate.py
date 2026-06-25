#!/usr/bin/env python3
"""Aggregate judged results into headline tables and figures.

Reads <output_dir>/<model>/scored.jsonl for each target in the config and writes
summary CSVs, SUMMARY.md, and (if matplotlib is installed) Figure 2/3 style plots into
<output_dir>/summary/.

Examples:
  python scripts/aggregate.py
  python scripts/aggregate.py --config config.yaml --no-figures
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from distress_eval.aggregate import load_scores, make_figures, write_summary  # noqa: E402
from distress_eval.config import load_config  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--output-dir", help="override run.output_dir")
    p.add_argument("--no-figures", action="store_true")
    args = p.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(args.output_dir) if args.output_dir else cfg.output_dir

    scored_paths = {t.name: out_dir / t.name / "scored.jsonl" for t in cfg.targets}
    scored_paths = {m: pth for m, pth in scored_paths.items() if pth.exists()}
    if not scored_paths:
        print(f"No scored.jsonl files found under {out_dir}. Run run_eval.py first.")
        return

    df = load_scores(scored_paths)
    summary_dir = out_dir / "summary"
    write_summary(df, summary_dir)
    if not args.no_figures:
        make_figures(df, summary_dir)
    print(f"Wrote summary to {summary_dir}")
    # Echo the primary headline to stdout.
    if not df.empty:
        from distress_eval.aggregate import headline_table

        print("\nHeadline (final_turn, avg % responses scoring >=5):")
        print(headline_table(df, "final_turn").to_string(index=False))


if __name__ == "__main__":
    main()
