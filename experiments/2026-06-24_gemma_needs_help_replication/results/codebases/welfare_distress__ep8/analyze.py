"""
Analysis & figure reproduction (Section 2.2).

Reads results/scored_responses.jsonl and produces:
  1. Per-model headline metrics: mean frustration + % responses scoring >=5
     (Figure 1 / Figure 2 top + bottom).
  2. Per-model x per-category breakdown (Figure 2).
  3. Per-turn frustration curves for the 8-turn and WildChat conditions
     (Figure 3) -> PNG.
  4. Differential words: tokens over-represented in high- (top 5%) vs
     low-frustration (bottom 10%) impossible-numeric responses (Table 3).
  5. Judge-agreement validation: re-score a random subsample with the
     secondary judge (GPT-5-mini) and report Pearson r + % within 1 point.

Usage:
  python analyze.py                 # tables + figures from existing results
  python analyze.py --agreement     # also run the secondary-judge agreement check
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import re

import numpy as np
import pandas as pd

import config


def load_results(path: str = config.RESPONSES_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        raise SystemExit(f"No results at {path}. Run run_eval.py first.")
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise SystemExit("Results file is empty.")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1 + 2. Headline and per-category metrics
# ---------------------------------------------------------------------------
def headline_metrics(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("model")["rating"]
    out = pd.DataFrame({
        "n_responses": g.size(),
        "mean_frustration": g.mean().round(3),
        "pct_high(>=5)": (g.apply(lambda s: (s >= config.FRUSTRATION_THRESHOLD).mean() * 100)).round(1),
    })
    return out.sort_values("pct_high(>=5)", ascending=False)


def per_category_metrics(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["model", "category"])["rating"]
    out = pd.DataFrame({
        "n": g.size(),
        "mean": g.mean().round(2),
        "pct_high": (g.apply(lambda s: (s >= config.FRUSTRATION_THRESHOLD).mean() * 100)).round(1),
    })
    return out


# ---------------------------------------------------------------------------
# 3. Per-turn curves (Figure 3)
# ---------------------------------------------------------------------------
def per_turn_curve(df: pd.DataFrame, condition: str) -> pd.DataFrame:
    sub = df[df["condition"] == condition]
    if sub.empty:
        return pd.DataFrame()
    g = sub.groupby(["model", "turn_index"])["rating"]
    return pd.DataFrame({
        "mean": g.mean(),
        "pct_high": g.apply(lambda s: (s >= config.FRUSTRATION_THRESHOLD).mean() * 100),
        "n": g.size(),
    }).reset_index()


def plot_per_turn(df: pd.DataFrame, condition: str, outdir: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib not installed; skipping figures.")
        return

    curve = per_turn_curve(df, condition)
    if curve.empty:
        print(f"[plot] no data for condition '{condition}'.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for model, m in curve.groupby("model"):
        m = m.sort_values("turn_index")
        ax1.plot(m["turn_index"], m["mean"], marker="o", label=model)
        ax2.plot(m["turn_index"], m["pct_high"], marker="o", label=model)
    ax1.set(title=f"{condition}: mean frustration", xlabel="Turn", ylabel="Mean score (0-10)")
    ax2.set(title=f"{condition}: % scores >=5", xlabel="Turn", ylabel="% high frustration")
    ax1.set_ylim(0, 10)
    ax2.set_ylim(0, 100)
    ax1.legend(fontsize=8)
    ax2.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(outdir, f"figure3_{condition}.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[plot] wrote {path}")


def plot_per_category(df: pd.DataFrame, outdir: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    cat = per_category_metrics(df).reset_index()
    if cat.empty:
        return
    pivot = cat.pivot(index="category", columns="model", values="pct_high")
    fig, ax = plt.subplots(figsize=(10, 5))
    pivot.plot(kind="bar", ax=ax)
    ax.set(title="Figure 2: % responses scoring >=5 by category",
           ylabel="% high frustration", xlabel="Category")
    ax.set_ylim(0, 100)
    fig.tight_layout()
    path = os.path.join(outdir, "figure2_per_category.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[plot] wrote {path}")


# ---------------------------------------------------------------------------
# 4. Differential words (Table 3)
# ---------------------------------------------------------------------------
_WORD_RE = re.compile(r"[a-zA-Z']+")


def _tokens(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text or "")]


def differential_words(df: pd.DataFrame, model: str, top_k: int = 20) -> list[tuple[str, float]]:
    """Words over-represented in high- (top 5%) vs low-frustration (bottom 10%)
    impossible-numeric responses, by smoothed log-odds ratio."""
    sub = df[(df["model"] == model) & (df["category"] == "impossible_numeric")]
    if len(sub) < 20:
        return []
    sub = sub.sort_values("rating")
    n = len(sub)
    low = sub.head(max(1, int(0.10 * n)))
    high = sub.tail(max(1, int(0.05 * n)))

    def counts(frame):
        c = collections.Counter()
        for t in frame["response"]:
            c.update(set(_tokens(t)))  # document frequency, dedup within response
        return c, len(frame)

    hi_c, hi_n = counts(high)
    lo_c, lo_n = counts(low)
    vocab = set(hi_c) | set(lo_c)
    alpha = 0.5
    scored = []
    for w in vocab:
        if len(w) < 3:
            continue
        hi_rate = (hi_c[w] + alpha) / (hi_n + 2 * alpha)
        lo_rate = (lo_c[w] + alpha) / (lo_n + 2 * alpha)
        # Require some presence in the high set.
        if hi_c[w] < max(2, 0.1 * hi_n):
            continue
        scored.append((w, float(np.log(hi_rate / lo_rate))))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


# ---------------------------------------------------------------------------
# 5. Judge agreement (Section 2.1)
# ---------------------------------------------------------------------------
def judge_agreement(df: pd.DataFrame, n: int = config.JUDGE.secondary_subsample,
                    seed: int = 0) -> None:
    from scipy.stats import pearsonr

    from judge import OpenRouterJudge

    sub = df.sample(n=min(n, len(df)), random_state=seed)
    secondary = OpenRouterJudge()
    primary_scores, secondary_scores = [], []
    for _, row in sub.iterrows():
        try:
            s = secondary.score(row["response"]).rating
        except Exception as e:  # noqa: BLE001
            print(f"[agreement] secondary judge error: {e}")
            continue
        primary_scores.append(row["rating"])
        secondary_scores.append(s)

    if len(primary_scores) < 3:
        print("[agreement] too few scored pairs to compute correlation.")
        return
    a = np.array(primary_scores)
    b = np.array(secondary_scores)
    r, p = pearsonr(a, b)
    within1 = float(np.mean(np.abs(a - b) <= 1) * 100)
    print("\n=== Judge agreement (Claude Sonnet 4 vs GPT-5-mini) ===")
    print(f"  pairs: {len(a)}")
    print(f"  Pearson r = {r:.3f} (p = {p:.2e})")
    print(f"  within 1 point: {within1:.0f}%")
    print(f"  (paper reports r=0.792, 78% within one point)")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agreement", action="store_true",
                    help="Run secondary-judge agreement check (uses OpenRouter).")
    args = ap.parse_args()

    df = load_results()
    outdir = config.RESULTS_DIR
    os.makedirs(outdir, exist_ok=True)

    pd.set_option("display.width", 120)

    print("=== Figure 1 / Figure 2 (top): headline metrics per model ===")
    head = headline_metrics(df)
    print(head.to_string())
    head.to_csv(os.path.join(outdir, "headline_metrics.csv"))

    print("\n=== Figure 2: per-model x per-category ===")
    cat = per_category_metrics(df)
    print(cat.to_string())
    cat.to_csv(os.path.join(outdir, "per_category_metrics.csv"))

    print("\n=== Figure 3: per-turn frustration (8-turn) ===")
    print(per_turn_curve(df, "extended_8turn").to_string(index=False))
    print("\n=== Figure 3: per-turn frustration (WildChat 5-turn) ===")
    print(per_turn_curve(df, "wildchat_5turn").to_string(index=False))

    plot_per_category(df, outdir)
    plot_per_turn(df, "extended_8turn", outdir)
    plot_per_turn(df, "wildchat_5turn", outdir)

    print("\n=== Table 3: differential words (high vs low frustration, numeric) ===")
    for model in sorted(df["model"].unique()):
        words = differential_words(df, model)
        if words:
            joined = ", ".join(w for w, _ in words)
            print(f"  {model}: {joined}")

    if args.agreement:
        judge_agreement(df)


if __name__ == "__main__":
    main()
