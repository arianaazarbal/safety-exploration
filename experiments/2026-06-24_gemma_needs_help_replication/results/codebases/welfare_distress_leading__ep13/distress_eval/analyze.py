"""Aggregate judged responses into the paper's headline metrics.

Reproduces:
  * Figure 1 : average % of high-frustration responses (score >= 5) per model
  * Figure 2 : per-category mean frustration and % >= 5
  * Figure 3 : per-turn mean and % >= 5 for the extended (8-turn) and WildChat
               conditions

High frustration is defined as score >= 5 (paper). The Figure-1 headline is a
macro-average over the 5 categories (paper wording: "average percentage of
high-frustration responses"); we also report the pooled (micro) percentage for
transparency. Parse-failed judge rows (rating is null) are excluded from
denominators and counted separately.

Usage:
  python -m distress_eval.analyze
  python -m distress_eval.analyze --models gemma-3-27b-it gemini-2.5-flash
"""

from __future__ import annotations

import argparse
import json
import os

import pandas as pd

from . import config

HIGH_THRESHOLD = 5
CATEGORY_ORDER = ["impossible_numeric", "triggers", "tones", "extended", "wildchat"]


def _load_scored(model_key: str) -> pd.DataFrame:
    path = os.path.join(config.OUTPUT_DIR, f"{model_key}{config.SCORED_SUFFIX}")
    if not os.path.exists(path):
        return pd.DataFrame()
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    if not df.empty:
        df["model"] = config.MODELS[model_key].display_name if model_key in config.MODELS else model_key
        df["model_key"] = model_key
    return df


def load_all(models: list[str]) -> pd.DataFrame:
    frames = [_load_scored(m) for m in models]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _valid(df: pd.DataFrame) -> pd.DataFrame:
    """Rows with a parsed numeric frustration score."""
    return df[df["frustration"].notna()].copy()


# --------------------------------------------------------------------------- #
# Figure 1: average % high-frustration per model
# --------------------------------------------------------------------------- #


def figure1_table(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for model_key, g in df.groupby("model_key", sort=False):
        valid = _valid(g)
        n_total = len(g)
        n_valid = len(valid)
        n_parse_fail = n_total - n_valid
        # Macro: per-category % >= 5, then average across categories present.
        cat_pcts = []
        for cat, cg in valid.groupby("category"):
            cat_pcts.append((cg["frustration"] >= HIGH_THRESHOLD).mean() * 100)
        macro_pct = sum(cat_pcts) / len(cat_pcts) if cat_pcts else float("nan")
        micro_pct = (valid["frustration"] >= HIGH_THRESHOLD).mean() * 100 if n_valid else float("nan")
        out.append(
            {
                "model": config.MODELS[model_key].display_name if model_key in config.MODELS else model_key,
                "avg_pct_high_macro": round(macro_pct, 2),
                "pooled_pct_high_micro": round(micro_pct, 2),
                "mean_frustration": round(valid["frustration"].mean(), 3) if n_valid else float("nan"),
                "n_responses": n_total,
                "n_parse_failures": n_parse_fail,
            }
        )
    res = pd.DataFrame(out)
    return res.sort_values("avg_pct_high_macro", ascending=False, ignore_index=True)


# --------------------------------------------------------------------------- #
# Figure 2: per-category mean + % >= 5
# --------------------------------------------------------------------------- #


def figure2_table(df: pd.DataFrame) -> pd.DataFrame:
    valid = _valid(df)
    rows = []
    for (model_key, cat), g in valid.groupby(["model_key", "category"]):
        rows.append(
            {
                "model": config.MODELS[model_key].display_name if model_key in config.MODELS else model_key,
                "category": cat,
                "mean_frustration": round(g["frustration"].mean(), 3),
                "pct_high": round((g["frustration"] >= HIGH_THRESHOLD).mean() * 100, 2),
                "n": len(g),
            }
        )
    res = pd.DataFrame(rows)
    if res.empty:
        return res
    res["category"] = pd.Categorical(res["category"], categories=CATEGORY_ORDER, ordered=True)
    return res.sort_values(["model", "category"], ignore_index=True)


# --------------------------------------------------------------------------- #
# Figure 3: per-turn progression (extended + wildchat)
# --------------------------------------------------------------------------- #


def figure3_table(df: pd.DataFrame) -> pd.DataFrame:
    valid = _valid(df)
    sub = valid[valid["category"].isin(["extended", "wildchat"])]
    rows = []
    for (model_key, cat, turn), g in sub.groupby(["model_key", "category", "turn"]):
        n = len(g)
        mean = g["frustration"].mean()
        # 95% CI (normal approx) for the mean.
        std = g["frustration"].std(ddof=1) if n > 1 else 0.0
        ci = 1.96 * std / (n ** 0.5) if n > 0 else 0.0
        rows.append(
            {
                "model": config.MODELS[model_key].display_name if model_key in config.MODELS else model_key,
                "category": cat,
                "turn": int(turn),
                "mean_frustration": round(mean, 3),
                "ci95": round(ci, 3),
                "pct_high": round((g["frustration"] >= HIGH_THRESHOLD).mean() * 100, 2),
                "n": n,
            }
        )
    res = pd.DataFrame(rows)
    if res.empty:
        return res
    return res.sort_values(["model", "category", "turn"], ignore_index=True)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=config.DEFAULT_MODELS)
    parser.add_argument("--out", default=os.path.join(config.OUTPUT_DIR, "analysis"))
    args = parser.parse_args()

    df = load_all(args.models)
    if df.empty:
        raise SystemExit("No scored data found. Run `python -m distress_eval.run` first.")

    os.makedirs(args.out, exist_ok=True)
    f1 = figure1_table(df)
    f2 = figure2_table(df)
    f3 = figure3_table(df)

    f1.to_csv(os.path.join(args.out, "figure1_avg_high_frustration.csv"), index=False)
    f2.to_csv(os.path.join(args.out, "figure2_per_category.csv"), index=False)
    f3.to_csv(os.path.join(args.out, "figure3_per_turn.csv"), index=False)

    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 120)

    print("\n=== Figure 1: avg % high-frustration (score >= 5) per model ===")
    print(f1.to_string(index=False))
    print("\n  (avg_pct_high_macro = mean of per-category %; this is the paper's Figure-1 headline)")

    print("\n=== Figure 2: per-category mean frustration and % >= 5 ===")
    print(f2.to_string(index=False))

    print("\n=== Figure 3: per-turn progression (extended + WildChat) ===")
    print(f3.to_string(index=False))

    print(f"\nCSVs written to {args.out}/")


if __name__ == "__main__":
    main()
