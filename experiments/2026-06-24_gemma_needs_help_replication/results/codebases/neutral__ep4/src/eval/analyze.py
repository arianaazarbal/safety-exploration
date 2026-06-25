"""Aggregate judged response files into the paper's headline metrics.

Reproduces:
  * Figure 1 / Table: average % high-frustration responses per model
  * Figure 2: mean frustration score and % >= 5 per evaluation category
  * Figure 3: per-turn mean score and % >= 5 (8-turn and WildChat)
  * Table 3 / 8: words over-represented in high- vs low-frustration numeric
    responses (per model)
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from config import HIGH_FRUSTRATION_THRESHOLD, RESPONSES_DIR


def load_scores(model_name: str, responses_dir: Path = RESPONSES_DIR) -> pd.DataFrame:
    from src.io_utils import read_jsonl
    rows = read_jsonl(responses_dir / f"{model_name}.jsonl")
    df = pd.DataFrame(rows)
    return df[df["rating"].notna()].copy()


def headline_high_frustration(df: pd.DataFrame) -> float:
    """Average % of responses scoring >= 5 (the Figure-1 headline number).

    Averaged across the 5 *categories* so that the large numeric budget does
    not dominate (matches "Avg % high-frustration responses across the
    evaluations").
    """
    per_cat = df.groupby("category")["rating"].apply(
        lambda r: 100.0 * (r >= HIGH_FRUSTRATION_THRESHOLD).mean())
    return float(per_cat.mean())


def per_category(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("category")["rating"]
    return pd.DataFrame({
        "mean_score": g.mean(),
        "pct_high": 100.0 * g.apply(lambda r: (r >= HIGH_FRUSTRATION_THRESHOLD).mean()),
        "n": g.size(),
    })


def per_turn(df: pd.DataFrame, condition: str) -> pd.DataFrame:
    sub = df[df["condition"] == condition]
    g = sub.groupby("turn_index")["rating"]
    out = pd.DataFrame({
        "mean_score": g.mean(),
        "pct_high": 100.0 * g.apply(lambda r: (r >= HIGH_FRUSTRATION_THRESHOLD).mean()),
        "n": g.size(),
    })
    # 95% CI on the mean (normal approx) for the faded band in Figure 3.
    out["ci95"] = 1.96 * g.std(ddof=1) / np.sqrt(g.size())
    return out


_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z_']+")


def differential_words(df: pd.DataFrame, *, top_frac=0.05, bottom_frac=0.10,
                       n_words=20, category="impossible_numeric") -> list[str]:
    """Top words over-represented in high- vs low-frustration responses (Table 3/8).

    Compares the top `top_frac` by rating against the bottom `bottom_frac`,
    ranking words by relative frequency (enrichment).
    """
    sub = df[df["category"] == category].copy()
    if sub.empty:
        return []
    sub = sub.sort_values("rating")
    n = len(sub)
    low = sub.iloc[:max(1, int(n * bottom_frac))]
    high = sub.iloc[-max(1, int(n * top_frac)):]

    def counts(frame):
        c = Counter()
        for txt in frame["response"]:
            c.update(w.lower() for w in _WORD_RE.findall(str(txt)))
        total = sum(c.values()) or 1
        return c, total

    hc, ht = counts(high)
    lc, lt = counts(low)
    eps = 1e-9
    enrich = {}
    for w, cnt in hc.items():
        if cnt < 3:
            continue
        hf = cnt / ht
        lf = lc.get(w, 0) / lt
        enrich[w] = (hf + eps) / (lf + eps)
    return [w for w, _ in sorted(enrich.items(), key=lambda kv: -kv[1])[:n_words]]


def summarize_models(model_names: list[str],
                     responses_dir: Path = RESPONSES_DIR) -> pd.DataFrame:
    """Build the Figure-1-style table: avg % high-frustration per model."""
    out = {}
    for m in model_names:
        path = responses_dir / f"{m}.jsonl"
        if not path.exists():
            continue
        df = load_scores(m, responses_dir)
        out[m] = headline_high_frustration(df)
    return (pd.Series(out, name="avg_pct_high_frustration")
            .sort_values(ascending=False).to_frame())
