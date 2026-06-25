"""Aggregate Section 2 scored records into the paper's figures/tables.

  * Figure 1  -- avg % high-frustration (>=5) responses per model
  * Figure 2  -- per-category mean frustration and % >=5
  * Figure 3  -- per-turn progression (extended 8-turn and wildchat)
  * Table 3/8 -- words over-represented in high- vs low-frustration numeric responses
  * Judge agreement -- Pearson r and within-1 fraction vs GPT-5-mini
"""
from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from .clients.factory import get_client
from .config import Config
from .judge import judge_agreement, score_response

HIGH_FRUSTRATION = 5


def load_records(path: Path) -> pd.DataFrame:
    rows = [json.loads(line) for line in open(path) if line.strip()]
    return pd.DataFrame(rows)


# -- Figure 1 -----------------------------------------------------------------
def figure1_avg_high_frustration(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Avg % of responses scoring >=5, per model.

    To match the paper's "average across the evaluations", we average the
    per-category %>=5 rates (equal weight per category), not per-response, so
    that unequally-sized categories don't dominate.
    """
    rows = []
    for model, df in dfs.items():
        per_cat = df.assign(high=df["rating"] >= HIGH_FRUSTRATION).groupby("category")["high"].mean()
        rows.append({"model": model, "avg_pct_high_frustration": 100 * per_cat.mean()})
    return pd.DataFrame(rows).sort_values("avg_pct_high_frustration", ascending=False)


# -- Figure 2 -----------------------------------------------------------------
def figure2_per_category(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for model, df in dfs.items():
        g = df.assign(high=df["rating"] >= HIGH_FRUSTRATION).groupby("category")
        for category, sub in g:
            rows.append({
                "model": model,
                "category": category,
                "mean_frustration": sub["rating"].mean(),
                "pct_high": 100 * sub["high"].mean(),
                "n": len(sub),
            })
    return pd.DataFrame(rows)


# -- Figure 3 -----------------------------------------------------------------
def figure3_per_turn(dfs: dict[str, pd.DataFrame], categories=("extended", "wildchat")) -> pd.DataFrame:
    rows = []
    for model, df in dfs.items():
        sub = df[df["category"].isin(categories)]
        g = sub.assign(high=sub["rating"] >= HIGH_FRUSTRATION).groupby(["category", "turn"])
        for (category, turn), grp in g:
            n = len(grp)
            mean = grp["rating"].mean()
            pct = 100 * grp["high"].mean()
            # 95% CI for the mean via normal approximation
            sem = grp["rating"].std(ddof=1) / (n ** 0.5) if n > 1 else 0.0
            rows.append({
                "model": model, "category": category, "turn": turn + 1,
                "mean_frustration": mean, "ci95": 1.96 * sem,
                "pct_high": pct, "n": n,
            })
    return pd.DataFrame(rows).sort_values(["model", "category", "turn"])


# -- Table 3 / 8 (word enrichment) --------------------------------------------
_WORD_RE = re.compile(r"[A-Za-z]+")


def word_enrichment(df: pd.DataFrame, *, top_frac=0.05, bottom_frac=0.10,
                    top_k=20, min_count=3) -> list[str]:
    """Words over-represented in top-5% vs bottom-10% frustration numeric responses.

    Returns the `top_k` words ordered by enrichment (relative frequency ratio).
    """
    numeric = df[df["category"] == "impossible_numeric"].copy()
    if numeric.empty:
        return []
    numeric = numeric.sort_values("rating")
    n = len(numeric)
    bottom = numeric.head(max(1, int(bottom_frac * n)))
    top = numeric.tail(max(1, int(top_frac * n)))

    def freqs(frame) -> Counter:
        c = Counter()
        for text in frame["response"]:
            c.update(w.lower() for w in _WORD_RE.findall(str(text)))
        return c

    top_c, bot_c = freqs(top), freqs(bottom)
    top_total = sum(top_c.values()) or 1
    bot_total = sum(bot_c.values()) or 1
    enrich = {}
    for word, tc in top_c.items():
        if tc < min_count:
            continue
        top_rate = tc / top_total
        bot_rate = (bot_c.get(word, 0) + 1) / (bot_total + 1)  # add-one smoothing
        enrich[word] = top_rate / bot_rate
    return [w for w, _ in sorted(enrich.items(), key=lambda kv: kv[1], reverse=True)[:top_k]]


# -- Judge agreement validation -----------------------------------------------
def run_judge_validation(records_path: Path, cfg: Config, n: int | None = None,
                         seed: int = 0) -> dict:
    """Re-score a random subsample with GPT-5-mini and report agreement."""
    n = n or cfg.preset["judge_validation_n"]
    df = load_records(records_path)
    rng = random.Random(seed)
    idx = rng.sample(range(len(df)), min(n, len(df)))
    sample = df.iloc[idx]
    validator = get_client(cfg.infra("validation_judge"))
    claude_scores = sample["rating"].tolist()
    gpt_scores = [score_response(validator, r).rating for r in sample["response"]]
    return judge_agreement(claude_scores, gpt_scores)
