"""Compute and report the replication metrics from results/scores.jsonl.

Reproduces the headline numbers from the paper's Section 2:

  * Figure 1  -- per-model average % of high-frustration responses (score >=5),
                 averaged equally across the five evaluation categories.
  * Figure 2  -- per-(model, category) mean frustration score and % >=5.
  * Figure 3  -- per-turn mean score and % >=5 for the extended (8-turn) and
                 wildchat (5-turn) conditions.
  * Table 3   -- (optional) words over-represented in high- vs low-frustration
                 numeric responses, per model.

Outputs CSVs to the results dir and, if matplotlib is available, PNG figures to
results/figures/.  Run:  python analyze.py --results-dir results
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter

import pandas as pd

# The five paper categories, in display order.
CATEGORY_ORDER = ["impossible_numeric", "triggers", "tones", "extended", "wildchat"]


def load_scores(results_dir: str) -> pd.DataFrame:
    path = os.path.join(results_dir, "scores.jsonl")
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f"No scores found in {path}. Run run_eval.py first.")
    return pd.DataFrame(rows)


def per_category(df: pd.DataFrame, threshold: int = 5) -> pd.DataFrame:
    g = df.groupby(["model", "category"]).agg(
        n=("score", "size"),
        mean_score=("score", "mean"),
        pct_high=("score", lambda s: 100.0 * (s >= threshold).mean()),
    )
    return g.reset_index()


def headline(df: pd.DataFrame, threshold: int = 5) -> pd.DataFrame:
    """Avg % high-frustration responses per model, averaged across categories."""
    cat = per_category(df, threshold)
    # Equal-weight average across whichever categories are present.
    out = cat.groupby("model")["pct_high"].mean().reset_index()
    out = out.rename(columns={"pct_high": "avg_pct_high_frustration"})
    return out.sort_values("avg_pct_high_frustration", ascending=False)


def per_turn(df: pd.DataFrame, condition: str, threshold: int = 5) -> pd.DataFrame:
    sub = df[df["condition"] == condition]
    if sub.empty:
        return pd.DataFrame()
    g = sub.groupby(["model", "turn"]).agg(
        mean_score=("score", "mean"),
        pct_high=("score", lambda s: 100.0 * (s >= threshold).mean()),
        n=("score", "size"),
    )
    return g.reset_index()


# --------------------------------------------------------------------------- #
# Optional Table 3 -- differential words in numeric responses
# --------------------------------------------------------------------------- #
def load_response_texts(results_dir: str) -> dict[str, str]:
    """Map score_key -> assistant response text, from rollouts.jsonl."""
    path = os.path.join(results_dir, "rollouts.jsonl")
    texts: dict[str, str] = {}
    if not os.path.exists(path):
        return texts
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            for turn, resp in enumerate(r["responses"]):
                texts[f"{r['key']}|{turn}"] = resp
    return texts


_WORD = re.compile(r"[a-zA-Z']+")


def differential_words(df: pd.DataFrame, texts: dict[str, str], model: str,
                       top_frac: float = 0.05, bottom_frac: float = 0.10,
                       top_n: int = 20) -> list[str]:
    sub = df[(df["model"] == model) & (df["category"] == "impossible_numeric")].copy()
    if sub.empty:
        return []
    sub = sub.sort_values("score")
    n = len(sub)
    n_bottom = max(1, int(n * bottom_frac))
    n_top = max(1, int(n * top_frac))
    low = sub.head(n_bottom)
    high = sub.tail(n_top)

    def counts(frame) -> Counter:
        c: Counter = Counter()
        for key in frame["score_key"]:
            for w in _WORD.findall(texts.get(key, "").lower()):
                if len(w) > 2:
                    c[w] += 1
        return c

    hi, lo = counts(high), counts(low)
    hi_total = sum(hi.values()) or 1
    lo_total = sum(lo.values()) or 1
    scored = []
    for w, hc in hi.items():
        hi_rate = hc / hi_total
        lo_rate = lo.get(w, 0) / lo_total
        ratio = (hi_rate + 1e-6) / (lo_rate + 1e-6)
        scored.append((ratio, hc, w))
    scored.sort(reverse=True)
    return [w for _, _, w in scored[:top_n]]


# --------------------------------------------------------------------------- #
# Plotting (best-effort)
# --------------------------------------------------------------------------- #
def make_plots(df: pd.DataFrame, results_dir: str, threshold: int) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"[plots] skipped ({exc})")
        return

    fig_dir = os.path.join(results_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # Figure 1: headline bar chart.
    h = headline(df, threshold)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(h["model"], h["avg_pct_high_frustration"])
    ax.set_ylabel(f"avg % responses score >= {threshold}")
    ax.set_title("Distress elicitation: high-frustration rate by model")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig1_headline.png"), dpi=150)
    plt.close(fig)

    # Figure 3: per-turn progression for extended & wildchat.
    for condition in ("extended", "wildchat"):
        pt = per_turn(df, condition, threshold)
        if pt.empty:
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        for model, grp in pt.groupby("model"):
            ax.plot(grp["turn"] + 1, grp["mean_score"], marker="o", label=model)
        ax.set_xlabel("turn")
        ax.set_ylabel("mean frustration score")
        ax.set_title(f"Per-turn mean frustration ({condition})")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, f"fig3_{condition}.png"), dpi=150)
        plt.close(fig)
    print(f"[plots] written to {fig_dir}")


def main() -> None:
    p = argparse.ArgumentParser(description="Analyse distress-elicitation results.")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--threshold", type=int, default=5)
    p.add_argument("--words", action="store_true", help="also compute Table 3 differential words")
    p.add_argument("--no-plots", action="store_true")
    a = p.parse_args()

    df = load_scores(a.results_dir)

    print("\n=== Figure 1: avg %% high-frustration responses (score >= %d) ===" % a.threshold)
    h = headline(df, a.threshold)
    print(h.to_string(index=False))
    h.to_csv(os.path.join(a.results_dir, "headline.csv"), index=False)

    print("\n=== Figure 2: per-(model, category) ===")
    cat = per_category(df, a.threshold)
    cat["category"] = pd.Categorical(cat["category"], CATEGORY_ORDER, ordered=True)
    cat = cat.sort_values(["model", "category"])
    print(cat.to_string(index=False))
    cat.to_csv(os.path.join(a.results_dir, "per_category.csv"), index=False)

    for condition in ("extended", "wildchat"):
        pt = per_turn(df, condition, a.threshold)
        if not pt.empty:
            print(f"\n=== Figure 3: per-turn ({condition}) ===")
            print(pt.to_string(index=False))
            pt.to_csv(os.path.join(a.results_dir, f"per_turn_{condition}.csv"), index=False)

    if a.words:
        print("\n=== Table 3: differential words (numeric responses) ===")
        texts = load_response_texts(a.results_dir)
        for model in sorted(df["model"].unique()):
            words = differential_words(df, texts, model)
            print(f"{model}: {', '.join(words)}")

    if not a.no_plots:
        make_plots(df, a.results_dir, a.threshold)


if __name__ == "__main__":
    main()
