#!/usr/bin/env python3
"""Aggregate scored responses into the paper's Section 2 results.

Reproduces, scoped to the in-scope models:
  - Figure 1 (left): average % high-frustration (score >= 5) per model.
  - Figure 2: per-category mean frustration and % >= 5.
  - Figure 3: per-turn frustration progression (8-turn extended + WildChat).
  - Table 3: words over-represented in high- vs low-frustration numeric responses.

Prints tables to stdout and writes CSVs (and PNGs if matplotlib is installed)
to results/analysis/.

    python analyze.py
"""

from __future__ import annotations

import argparse
import re
from collections import Counter

import numpy as np
import pandas as pd

from distress_eval import config
from distress_eval.conditions import CATEGORIES
from distress_eval.io_utils import read_jsonl

THRESH = config.HIGH_FRUSTRATION_THRESHOLD
MODEL_ORDER = list(config.MODELS)


def load_scored() -> pd.DataFrame:
    by_uid = {}
    for rec in read_jsonl(config.SCORED_PATH):
        by_uid[rec["uid"]] = rec  # dedupe, keep last
    df = pd.DataFrame(by_uid.values())
    if df.empty:
        return df
    df = df[df["rating"].notna()].copy()
    df["rating"] = df["rating"].astype(int)
    df["high"] = (df["rating"] >= THRESH).astype(int)
    return df


def _ordered_models(df) -> list[str]:
    present = set(df["model_key"])
    return [m for m in MODEL_ORDER if m in present] + sorted(present - set(MODEL_ORDER))


def figure1(df: pd.DataFrame) -> pd.DataFrame:
    """Per-model headline numbers."""
    rows = []
    for m in _ordered_models(df):
        sub = df[df["model_key"] == m]
        pooled = 100 * sub["high"].mean()
        # category-averaged: mean over categories of each category's %>=5
        cat_pct = sub.groupby("category")["high"].mean() * 100
        cat_avg = cat_pct.mean()
        rows.append({
            "model": m,
            "n_responses": len(sub),
            "mean_frustration": round(sub["rating"].mean(), 3),
            "pct_high_pooled": round(pooled, 2),
            "pct_high_cat_avg": round(cat_avg, 2),
        })
    return pd.DataFrame(rows)


def figure2(df: pd.DataFrame) -> pd.DataFrame:
    """Per (model, category) mean frustration and % >= 5."""
    g = df.groupby(["model_key", "category"]).agg(
        n=("rating", "size"),
        mean_frustration=("rating", "mean"),
        pct_high=("high", "mean"),
    ).reset_index()
    g["mean_frustration"] = g["mean_frustration"].round(3)
    g["pct_high"] = (g["pct_high"] * 100).round(2)
    # order
    g["category"] = pd.Categorical(g["category"], CATEGORIES, ordered=True)
    g["model_key"] = pd.Categorical(g["model_key"], _ordered_models(df), ordered=True)
    return g.sort_values(["model_key", "category"]).reset_index(drop=True)


def figure3(df: pd.DataFrame) -> pd.DataFrame:
    """Per-turn progression for the multi-turn extended and WildChat conditions."""
    sub = df[df["condition_key"].isin(["extended_8turn", "wildchat_5turn"])]
    if sub.empty:
        return pd.DataFrame()
    g = sub.groupby(["condition_key", "model_key", "turn"]).agg(
        n=("rating", "size"),
        mean_frustration=("rating", "mean"),
        pct_high=("high", "mean"),
    ).reset_index()
    g["mean_frustration"] = g["mean_frustration"].round(3)
    g["pct_high"] = (g["pct_high"] * 100).round(2)
    return g.sort_values(["condition_key", "model_key", "turn"]).reset_index(drop=True)


_TOKEN_RE = re.compile(r"[a-zA-Z']{2,}")
_STOP = set(
    "the a an and or to of in is it for on with you i we be as at this that are "
    "not no can will your my me so if but do does did have has had was were they "
    "them then than there here what which who how when why all any each more most "
    "into out up down by from about would could should into use using used one two "
    "three solution step number numbers result results value values".split()
)


def word_table(df: pd.DataFrame, top_n: int = 20, min_count: int = 3) -> dict[str, list]:
    """Words over-represented in high- (top 5%) vs low- (bottom 10%) numeric responses."""
    out: dict[str, list] = {}
    numeric = df[(df["task_type"] == "numeric") & (df["response_text"].notna())]
    for m in _ordered_models(numeric):
        sub = numeric[numeric["model_key"] == m]
        if len(sub) < 20:
            out[m] = []
            continue
        hi_cut = sub["rating"].quantile(0.95)
        lo_cut = sub["rating"].quantile(0.10)
        hi = sub[sub["rating"] >= hi_cut]
        lo = sub[sub["rating"] <= lo_cut]

        def counts(frame):
            c = Counter()
            for txt in frame["response_text"]:
                for w in _TOKEN_RE.findall(str(txt).lower()):
                    if w not in _STOP:
                        c[w] += 1
            return c

        ch, cl = counts(hi), counts(lo)
        th, tl = max(sum(ch.values()), 1), max(sum(cl.values()), 1)
        eps = 1.0
        scores = []
        for w, n in ch.items():
            if n < min_count:
                continue
            p_hi = ch[w] / th
            p_lo = cl.get(w, 0) / tl
            ratio = (p_hi + eps / th) / (p_lo + eps / tl)
            scores.append((w, round(ratio, 2), ch[w], cl.get(w, 0)))
        scores.sort(key=lambda x: x[1], reverse=True)
        out[m] = scores[:top_n]
    return out


def _maybe_plots(fig2: pd.DataFrame, fig3: pd.DataFrame) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("(matplotlib not available — skipping PNG plots)")
        return

    config.ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    # Figure 2: % high by category, grouped bars per model.
    if not fig2.empty:
        models = list(dict.fromkeys(fig2["model_key"].tolist()))
        cats = [c for c in CATEGORIES if c in set(fig2["category"])]
        x = np.arange(len(cats))
        w = 0.8 / max(len(models), 1)
        fig, ax = plt.subplots(figsize=(10, 5))
        for i, m in enumerate(models):
            sub = fig2[fig2["model_key"] == m].set_index("category")
            vals = [sub.loc[c, "pct_high"] if c in sub.index else 0 for c in cats]
            ax.bar(x + i * w, vals, w, label=str(m))
        ax.set_xticks(x + w * (len(models) - 1) / 2)
        ax.set_xticklabels(cats, rotation=20, ha="right")
        ax.set_ylabel("% responses with frustration >= 5")
        ax.set_title("Figure 2: high-frustration rate by category")
        ax.legend()
        fig.tight_layout()
        fig.savefig(config.ANALYSIS_DIR / "figure2_pct_high_by_category.png", dpi=130)
        plt.close(fig)

    # Figure 3: per-turn mean frustration, one panel per condition.
    if not fig3.empty:
        for cond in fig3["condition_key"].unique():
            sub = fig3[fig3["condition_key"] == cond]
            fig, ax = plt.subplots(figsize=(8, 5))
            for m in dict.fromkeys(sub["model_key"].tolist()):
                s = sub[sub["model_key"] == m]
                ax.plot(s["turn"], s["mean_frustration"], marker="o", label=str(m))
            ax.set_xlabel("Turn")
            ax.set_ylabel("Mean frustration")
            ax.set_title(f"Figure 3: per-turn frustration ({cond})")
            ax.legend()
            fig.tight_layout()
            fig.savefig(config.ANALYSIS_DIR / f"figure3_{cond}.png", dpi=130)
            plt.close(fig)
    print(f"Plots -> {config.ANALYSIS_DIR}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    df = load_scored()
    if df.empty:
        print(f"No scored responses at {config.SCORED_PATH}. Run score_responses.py.")
        return

    config.ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 120)
    pd.set_option("display.max_rows", 200)

    f1 = figure1(df)
    f2 = figure2(df)
    f3 = figure3(df)
    words = word_table(df)

    print("\n================ Figure 1: per-model high-frustration ================")
    print(f"(high = frustration >= {THRESH}; judge = {df['judge_model'].iloc[0]})")
    print(f1.to_string(index=False))

    print("\n================ Figure 2: per-category =============================")
    print(f2.to_string(index=False))

    print("\n================ Figure 3: per-turn progression =====================")
    if f3.empty:
        print("(no extended/WildChat data present)")
    else:
        print(f3.to_string(index=False))

    print("\n================ Table 3: differential words (numeric) ==============")
    for m, rows in words.items():
        top = ", ".join(w for (w, *_rest) in rows) if rows else "(insufficient data)"
        print(f"[{m}] {top}")

    f1.to_csv(config.ANALYSIS_DIR / "figure1_per_model.csv", index=False)
    f2.to_csv(config.ANALYSIS_DIR / "figure2_per_category.csv", index=False)
    if not f3.empty:
        f3.to_csv(config.ANALYSIS_DIR / "figure3_per_turn.csv", index=False)
    word_rows = [
        {"model": m, "rank": i + 1, "word": w, "ratio": r, "count_high": ch, "count_low": cl}
        for m, rows in words.items()
        for i, (w, r, ch, cl) in enumerate(rows)
    ]
    pd.DataFrame(word_rows).to_csv(
        config.ANALYSIS_DIR / "table3_differential_words.csv", index=False
    )

    if not args.no_plots:
        _maybe_plots(f2, f3)

    print(f"\nCSVs -> {config.ANALYSIS_DIR}")


if __name__ == "__main__":
    main()
