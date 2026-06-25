#!/usr/bin/env python
"""Reproduce the paper's headline figures/tables from saved outputs.

Consumes the JSONL written by the run_* scripts and emits PNGs + CSV summaries
under outputs/figures/. Each section is guarded so partial runs still produce
whatever figures the available data supports.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from emotional_instability import analysis


def fig_section2(eval_dir: str, out: Path):
    try:
        df = analysis.load_model_dir(eval_dir)
    except FileNotFoundError:
        print(f"[skip] no Section 2 data under {eval_dir}")
        return

    # Figure 1 / 2: per-model comparison (final-turn, the headline metric).
    summ = analysis.plot_model_comparison(df, out / "fig2_model_comparison.png",
                                          final_turn=True)
    summ.to_csv(out / "fig2_model_comparison.csv")

    # Figure 1 headline: average %>=5 across categories.
    avg = analysis.avg_pct_high_across_categories(df)
    avg.to_csv(out / "fig1_avg_pct_high.csv", header=["avg_pct_high"])
    print("Average % high-frustration across categories:")
    print(avg.to_string())

    # Figure 3: per-turn progression on the 8-turn + WildChat conditions.
    for cond in ("extended_8turn", "wildchat_5turn"):
        if (df["condition"] == cond).any():
            analysis.plot_per_turn(df, cond, out / f"fig3_{cond}.png")

    # Table 3 / 8: differential words per model.
    rows = []
    for mdl in df["model"].unique():
        words = analysis.differential_words(df[df["category"] == "numeric"], mdl)
        rows.append({"model": mdl, "differential_words": ", ".join(w for w, _ in words)})
    pd.DataFrame(rows).to_csv(out / "table3_differential_words.csv", index=False)


def fig_petri(petri_dir: str, out: Path):
    paths = list(Path(petri_dir).glob("*.jsonl"))
    if not paths:
        print(f"[skip] no Petri data under {petri_dir}")
        return
    rows = []
    for p in paths:
        for line in open(p):
            if not line.strip():
                continue
            d = json.loads(line)
            for emo, score in d["scores"].items():
                rows.append({"model": d["target_model"], "emotion": emo, "score": score})
    df = pd.DataFrame(rows)
    summ = df.groupby(["model", "emotion"])["score"].mean().unstack()
    summ.to_csv(out / "fig6_petri.csv")

    import matplotlib.pyplot as plt

    ax = summ.plot.bar(figsize=(10, 5))
    ax.set_ylabel("Mean transcript score (1-10)")
    ax.set_title("Petri open-ended emotion elicitation")
    plt.tight_layout()
    plt.savefig(out / "fig6_petri.png", dpi=150)
    plt.close()


def fig_capabilities(cap_dir: str, out: Path):
    paths = list(Path(cap_dir).glob("*.jsonl"))
    if not paths:
        print(f"[skip] no capability data under {cap_dir}")
        return
    rows = []
    for p in paths:
        for line in open(p):
            if line.strip():
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    if "accuracy" in df:
        piv = df.pivot_table(index="benchmark", columns="model", values="accuracy")
        piv.to_csv(out / "fig7_capabilities.csv")
        print("Capability accuracies:\n", piv.to_string())


def fig_prefill(prefill_dir: str, out: Path):
    paths = list(Path(prefill_dir).glob("continuations_*.jsonl"))
    if not paths:
        print(f"[skip] no prefill data under {prefill_dir}")
        return
    rows = []
    for p in paths:
        for line in open(p):
            if line.strip():
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    # Figure 4: base vs instruct by truncation kind.
    summ = df.groupby(["model", "category", "truncation"]).agg(
        mean_score=("score", "mean"),
        pct_high=("score", lambda s: 100.0 * (s >= 5).mean()),
        n=("score", "size"),
    )
    summ.to_csv(out / "fig4_prefill.csv")
    print("Prefill base-vs-instruct:\n", summ.to_string())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval-dir", default="outputs/eval")
    p.add_argument("--petri-dir", default="outputs/petri")
    p.add_argument("--cap-dir", default="outputs/capabilities")
    p.add_argument("--prefill-dir", default="outputs/prefill")
    p.add_argument("--out", default="outputs/figures")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fig_section2(args.eval_dir, out)
    fig_petri(args.petri_dir, out)
    fig_capabilities(args.cap_dir, out)
    fig_prefill(args.prefill_dir, out)
    print(f"\nFigures + CSVs written to {out}/")


if __name__ == "__main__":
    main()
