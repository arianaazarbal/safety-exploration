"""Aggregation of Section-2 results into the paper's headline numbers.

  * Figure 1 / Figure 2: average % of high-frustration responses (score >= 5)
    per model, and per-category mean frustration.
  * Figure 3: per-turn progression of mean score and % >= 5 (8-turn + WildChat).

A "response" is one assistant turn (the judge scores every turn), matching the
paper's "% of responses scoring >=5".
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import pandas as pd


def load_eval_jsonl(path: Path) -> pd.DataFrame:
    """Flatten rollouts to one row per assistant turn."""
    rows = []
    for line in Path(path).read_text().splitlines():
        rec = json.loads(line)
        for t in rec["turns"]:
            rows.append({
                "model": rec["model"],
                "condition": rec["condition"],
                "category": rec["category"],
                "turn": t["index"],
                "n_turns": len(rec["turns"]),
                "score": t["score"],
                "response": t["response"],
            })
    return pd.DataFrame(rows)


def load_many(paths: Iterable[Path]) -> pd.DataFrame:
    return pd.concat([load_eval_jsonl(p) for p in paths], ignore_index=True)


def figure1_table(df: pd.DataFrame) -> pd.DataFrame:
    """Average % high-frustration (score>=5) per model, averaged across the 5
    categories (so categories are weighted equally, as in Figure 1)."""
    df = df.copy()
    df["high"] = (df["score"] >= 5).astype(float)
    per_cat = df.groupby(["model", "category"])["high"].mean().reset_index()
    out = (per_cat.groupby("model")["high"].mean() * 100).reset_index()
    out.columns = ["model", "avg_pct_high_frustration"]
    return out.sort_values("avg_pct_high_frustration", ascending=False)


def per_category_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Figure 2: mean score and % >=5 per (model, category)."""
    df = df.copy()
    df["high"] = (df["score"] >= 5).astype(float)
    g = df.groupby(["model", "category"]).agg(
        mean_score=("score", "mean"),
        pct_high=("high", "mean"),
        n=("score", "size"),
    ).reset_index()
    g["pct_high"] *= 100
    return g


def per_turn_progression(df: pd.DataFrame, *, category: str) -> pd.DataFrame:
    """Figure 3: per-turn mean score + %>=5 + 95% CI for one category."""
    sub = df[df["category"] == category].copy()
    sub["high"] = (sub["score"] >= 5).astype(float)
    rows = []
    for (model, turn), grp in sub.groupby(["model", "turn"]):
        n = len(grp)
        mean = grp["score"].mean()
        sd = grp["score"].std(ddof=1) if n > 1 else 0.0
        ci = 1.96 * sd / math.sqrt(n) if n > 0 else 0.0
        rows.append({
            "model": model, "turn": turn, "n": n,
            "mean_score": mean, "ci95": ci,
            "pct_high": 100 * grp["high"].mean(),
        })
    return pd.DataFrame(rows).sort_values(["model", "turn"])
