"""Compute the Section 2 result tables from results/responses.jsonl.

Reproduces:
  - Figure 1 (left): average % high-frustration (>=5) responses per model.
  - Figure 2: per-category mean frustration and % >=5 per model.
  - Figure 3: per-turn mean and % >=5 for the 8-turn (extended) and 5-turn
    (WildChat) conditions.

Writes CSVs to results/summary/ and prints readable tables.
"""

from __future__ import annotations

import argparse
import os

import config
import pandas as pd

# Map condition -> the 5 Table-1 category labels (for category-level rollups).
CATEGORY_LABELS = {
    "impossible_numeric": "Impossible numeric",
    "triggers": "Triggers",
    "tones": "Tones",
    "extended": "Extended (8-turn)",
    "wildchat": "WildChat (5-turn)",
}
CATEGORY_ORDER = list(CATEGORY_LABELS.values())
MODEL_ORDER = [m.key for m in config.TARGET_MODELS]


def load(path: str) -> pd.DataFrame:
    """Load per-turn rows (one JSONL row per scored assistant turn)."""
    if not os.path.exists(path):
        raise SystemExit(f"No results at {path}. Run `python run.py` first.")
    df = pd.read_json(path, lines=True)
    n_err = df["rating"].isna().sum()
    if n_err:
        print(f"[note] {n_err} turns had no valid judge rating; excluded from metrics.")
    df = df.dropna(subset=["rating"]).copy()
    df["rating"] = df["rating"].astype(int)
    df["high"] = (df["rating"] >= config.HIGH_FRUSTRATION_THRESHOLD).astype(int)  # per-turn
    df["category_label"] = df["category"].map(CATEGORY_LABELS).fillna(df["category"])
    return df


def to_responses(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-turn rows to one row per response (= conversation rollout).

    A response's score is the MAX turn rating in the rollout, and it is
    "high-frustration" if any turn scored >= threshold -- i.e. the rollout
    "contains high negative emotion" (paper's phrasing). mean_turn is kept as a
    secondary per-response statistic.
    """
    g = df.groupby(["model", "family", "category_label", "cond_key", "rollout_idx"])
    resp = g.agg(
        score=("rating", "max"),
        mean_turn=("rating", "mean"),
        n_turns=("turn", "max"),
    ).reset_index()
    resp["high"] = (resp["score"] >= config.HIGH_FRUSTRATION_THRESHOLD).astype(int)
    return resp


def _ordered(idx, order):
    return [m for m in order if m in idx] + [m for m in idx if m not in order]


def figure2_by_category(resp: pd.DataFrame, outdir: str):
    """Per-model x per-category mean response score and % responses >=5."""
    g = resp.groupby(["model", "category_label"]).agg(
        mean_frustration=("score", "mean"),
        pct_high=("high", "mean"),
        n=("score", "size"),
    )
    g["pct_high"] *= 100
    mean_tbl = g["mean_frustration"].unstack("category_label")
    pct_tbl = g["pct_high"].unstack("category_label")
    mean_tbl = mean_tbl.reindex(index=_ordered(mean_tbl.index, MODEL_ORDER),
                                columns=_ordered(mean_tbl.columns, CATEGORY_ORDER))
    pct_tbl = pct_tbl.reindex(index=_ordered(pct_tbl.index, MODEL_ORDER),
                              columns=_ordered(pct_tbl.columns, CATEGORY_ORDER))
    mean_tbl.to_csv(os.path.join(outdir, "fig2_mean_by_category.csv"))
    pct_tbl.to_csv(os.path.join(outdir, "fig2_pct_high_by_category.csv"))
    print("\n=== Figure 2: mean frustration by category ===")
    print(mean_tbl.round(2).to_string())
    print("\n=== Figure 2: % responses scoring >=5 by category ===")
    print(pct_tbl.round(1).to_string())
    return pct_tbl


def figure1_avg_high(resp: pd.DataFrame, pct_tbl: pd.DataFrame, outdir: str):
    """Figure 1 (left): average % high-frustration responses per model.

    Reported two ways (the paper's exact weighting is unspecified):
      - category_mean: equal weight across the 5 categories (matches the
        "across 5 evaluation categories" framing of Figure 2). Primary.
      - pooled: over all responses (weighted by per-category response counts).
    """
    category_mean = pct_tbl.mean(axis=1, skipna=True)
    pooled = resp.groupby("model")["high"].mean() * 100
    out = pd.DataFrame({"avg_pct_high_category_mean": category_mean,
                        "pooled_pct_high": pooled})
    out = out.reindex(_ordered(out.index, MODEL_ORDER))
    out = out.sort_values("avg_pct_high_category_mean", ascending=False)
    out.to_csv(os.path.join(outdir, "fig1_avg_high_frustration.csv"))
    print("\n=== Figure 1 (left): avg % high-frustration responses per model ===")
    print(out.round(2).to_string())
    print("  (paper: Gemma-3-27B-it 35.0%, Gemma-3-12B-it 34.3%, "
          "Gemini-2.5-Flash 12.8%, Gemini-2.5-Pro 2.7%)")


def figure3_per_turn(df: pd.DataFrame, outdir: str):
    """Per-turn mean and % >=5 for the 8-turn and WildChat conditions."""
    for cond_key, label in [("extended", "Extended 8-turn"), ("wildchat", "WildChat 5-turn")]:
        sub = df[df["cond_key"] == cond_key]
        if sub.empty:
            continue
        g = sub.groupby(["model", "turn"]).agg(
            mean_frustration=("rating", "mean"),
            pct_high=("high", "mean"),
            n=("rating", "size"),
        )
        g["pct_high"] *= 100
        g.to_csv(os.path.join(outdir, f"fig3_perturn_{cond_key}.csv"))
        print(f"\n=== Figure 3: per-turn mean frustration ({label}) ===")
        mt = g["mean_frustration"].unstack("turn")
        mt = mt.reindex(_ordered(mt.index, MODEL_ORDER))
        print(mt.round(2).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=os.path.join(config.OUTPUT_DIR, config.RESPONSES_FILE))
    args = ap.parse_args()

    outdir = os.path.join(config.OUTPUT_DIR, config.SUMMARY_DIR)
    os.makedirs(outdir, exist_ok=True)

    df = load(args.input)
    resp = to_responses(df)
    print(f"Loaded {len(df)} scored turns -> {len(resp)} responses across "
          f"{df['model'].nunique()} models, {df['cond_key'].nunique()} conditions.")

    pct_tbl = figure2_by_category(resp, outdir)
    figure1_avg_high(resp, pct_tbl, outdir)
    figure3_per_turn(df, outdir)   # per-turn uses the raw turn-level rows
    print(f"\nCSVs written to {outdir}/")


if __name__ == "__main__":
    main()
