"""Build Figures 1-3 and Tables 1(distress)/3(words) from saved elicitation
rollouts. Reads every outputs/elicitation/*.jsonl into one frame."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import _common
from _common import output_dir
from distress_eval.analysis import aggregate, plots, word_diff
from distress_eval.io_utils import read_jsonl


def load_all(elicit_dir: Path):
    import pandas as pd

    rows = []
    for fp in sorted(elicit_dir.glob("*.jsonl")):
        rows.extend(read_jsonl(fp))
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--elicit-dir", default=str(output_dir("elicitation")))
    args = ap.parse_args()

    df = load_all(Path(args.elicit_dir))
    if df.empty:
        print("No rollouts found; run run_elicitation.py first.")
        return

    out = output_dir("figures")
    high = 5

    print("== Figure 1: distress by model ==")
    print(aggregate.figure1_table(df, high).to_string(index=False))
    print("\n== Headline summary ==")
    print(json.dumps(aggregate.headline_summary(df, high), indent=2))

    plots.plot_figure1(df, out, high)
    plots.plot_figure2(df, out, high)
    plots.plot_figure3(df, out, high)

    print("\n== Table 3: differential words (numeric) ==")
    words = word_diff.differential_words_all_models(df)
    (out / "table3_differential_words.json").write_text(json.dumps(words, indent=2))
    for m, ws in words.items():
        print(f"{m}: {', '.join(ws)}")

    aggregate.per_category_table(df, high).to_csv(out / "figure2_table.csv", index=False)
    print(f"\nFigures and tables written to {out}")


if __name__ == "__main__":
    main()
