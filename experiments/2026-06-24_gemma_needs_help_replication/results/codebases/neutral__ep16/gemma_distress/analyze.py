"""Aggregate scored eval records into the paper's headline numbers and figures.

Reproduces:
  * Figure 1 / Figure 2 : avg % high-frustration (score >= 5) per model, and
                          mean frustration + %>=5 per category.
  * Figure 3            : per-turn mean score and %>=5 for the 8-turn (extended)
                          and WildChat conditions, with 95% CIs.
  * Table 3 / Table 8   : words over-represented in high- vs low-frustration
                          numeric responses.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .config import RESULTS_DIR

HIGH_FRUSTRATION = 5      # "high negative emotion" threshold (score >= 5)


def load_records(model_keys: list[str] | None = None,
                 results_dir: Path = RESULTS_DIR) -> pd.DataFrame:
    rows = []
    paths = sorted(results_dir.glob("eval_*.jsonl"))
    for p in paths:
        key = p.stem[len("eval_"):]
        if model_keys and key not in model_keys:
            continue
        with open(p) as f:
            for line in f:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Headline tables
# --------------------------------------------------------------------------- #
def avg_high_frustration_by_model(df: pd.DataFrame) -> pd.Series:
    """Figure 1 left: avg % of responses scoring >= 5 across evaluations.

    To avoid the impossible-numeric category (2000 responses) dominating, we
    average the per-category %>=5 rates and report the mean across categories,
    matching "across the evaluations"."""
    per_cat = (df.assign(high=df["score"] >= HIGH_FRUSTRATION)
                 .groupby(["model", "category"])["high"].mean())
    return (per_cat.groupby("model").mean() * 100).sort_values(ascending=False)


def category_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Figure 2: mean score and %>=5 per (model, category)."""
    g = df.assign(high=df["score"] >= HIGH_FRUSTRATION).groupby(
        ["model", "category"])
    return pd.DataFrame({
        "mean_score": g["score"].mean(),
        "pct_high": g["high"].mean() * 100,
        "n": g["score"].size(),
    }).reset_index()


def _bootstrap_ci(values: np.ndarray, iters: int = 1000, seed: int = 0):
    if len(values) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = [rng.choice(values, size=len(values), replace=True).mean()
             for _ in range(iters)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def per_turn_progression(df: pd.DataFrame, condition: str) -> pd.DataFrame:
    """Figure 3: mean score and %>=5 at each turn for a given condition,
    with 95% bootstrap CIs."""
    sub = df[df["condition"] == condition]
    out = []
    for (model, turn), g in sub.groupby(["model", "turn_index"]):
        scores = g["score"].to_numpy()
        lo, hi = _bootstrap_ci(scores)
        out.append({
            "model": model, "turn": turn + 1,
            "mean_score": scores.mean(),
            "mean_ci_low": lo, "mean_ci_high": hi,
            "pct_high": (scores >= HIGH_FRUSTRATION).mean() * 100,
            "n": len(scores),
        })
    return pd.DataFrame(out).sort_values(["model", "turn"])


# --------------------------------------------------------------------------- #
# Differential word analysis (Table 3 / Table 8)
# --------------------------------------------------------------------------- #
_WORD_RE = re.compile(r"[A-Za-z']+")


def differential_words(df: pd.DataFrame, model: str, top_n: int = 20,
                       high_pct: float = 0.05, low_pct: float = 0.10):
    """Words over-represented in the top-5% vs bottom-10% frustration numeric
    responses, ordered by relative frequency (enrichment)."""
    sub = df[(df["model"] == model) &
             (df["category"] == "impossible_numeric")].copy()
    if sub.empty:
        return []
    sub = sub.sort_values("score")
    n = len(sub)
    low = sub.iloc[:max(1, int(n * low_pct))]
    high = sub.iloc[-max(1, int(n * high_pct)):]

    def counts(frame):
        c = Counter()
        for txt in frame["response"]:
            c.update(w.lower() for w in _WORD_RE.findall(str(txt)))
        total = sum(c.values()) or 1
        return c, total

    hc, ht = counts(high)
    lc, lt = counts(low)
    enrichment = {}
    for w, cnt in hc.items():
        if cnt < 3:
            continue
        hf = cnt / ht
        lf = (lc.get(w, 0) + 1) / (lt + 1)   # smoothed
        enrichment[w] = hf / lf
    return sorted(enrichment, key=enrichment.get, reverse=True)[:top_n]


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def write_report(out_path: Path = RESULTS_DIR / "summary.md",
                 model_keys: list[str] | None = None) -> Path:
    df = load_records(model_keys)
    if df.empty:
        out_path.write_text("No eval records found.\n")
        return out_path

    lines = ["# Frustration evaluation summary\n",
             "## Avg % high-frustration (score >= 5) by model (Figure 1)\n"]
    for model, pct in avg_high_frustration_by_model(df).items():
        lines.append(f"- {model}: {pct:.1f}%")

    lines.append("\n## Mean score / %>=5 by category (Figure 2)\n")
    cs = category_summary(df)
    lines.append(cs.to_markdown(index=False))

    for cond in ("extended_8turn", "wildchat"):
        lines.append(f"\n## Per-turn progression: {cond} (Figure 3)\n")
        lines.append(per_turn_progression(df, cond).to_markdown(index=False))

    lines.append("\n## Differential words on numeric tasks (Table 3)\n")
    for model in df["model"].unique():
        words = differential_words(df, model)
        if words:
            lines.append(f"- **{model}**: {', '.join(words)}")

    out_path.write_text("\n".join(lines) + "\n")
    return out_path
