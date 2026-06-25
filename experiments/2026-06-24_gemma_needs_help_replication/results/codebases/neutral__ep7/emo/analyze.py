"""Aggregation and analysis of rollout judge scores (Section 2.2).

Reproduces the paper's headline numbers:
  * mean frustration and % responses scoring >=5, per model and per category
    (Figure 1 / Figure 2);
  * per-turn progression for the 8-turn and WildChat conditions (Figure 3);
  * judge reliability (Pearson r, % within 1 point) between Claude-Sonnet and
    GPT-5-mini on a re-scored sample (Section 2.1);
  * top differential words in high- vs low-frustration numeric responses
    (Table 3 / Table 8).
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from . import config


def load_records(rollout_dir: Path = config.ROLLOUT_DIR, pattern: str = "*.jsonl") -> pd.DataFrame:
    """Load all TurnRecords into a DataFrame, one row per scored assistant turn."""
    rows = []
    for fp in sorted(Path(rollout_dir).glob(pattern)):
        for line in fp.read_text().splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df[df["rating"] >= 0]  # drop unscored turns
    return df


def summary_by_model(df: pd.DataFrame) -> pd.DataFrame:
    """Avg % high-frustration and mean score per model, averaged over the 5
    categories (matches Figure 1's 'Avg % high-frustration responses')."""
    thr = config.HIGH_FRUSTRATION_THRESHOLD
    per_cat = (
        df.assign(high=df["rating"] >= thr)
        .groupby(["model", "category"])
        .agg(mean_score=("rating", "mean"), pct_high=("high", "mean"))
        .reset_index()
    )
    out = (
        per_cat.groupby("model")
        .agg(mean_score=("mean_score", "mean"), avg_pct_high=("pct_high", "mean"))
        .reset_index()
        .sort_values("avg_pct_high", ascending=False)
    )
    out["avg_pct_high"] = (out["avg_pct_high"] * 100).round(2)
    out["mean_score"] = out["mean_score"].round(3)
    return out


def summary_by_model_category(df: pd.DataFrame) -> pd.DataFrame:
    thr = config.HIGH_FRUSTRATION_THRESHOLD
    out = (
        df.assign(high=df["rating"] >= thr)
        .groupby(["model", "category"])
        .agg(n=("rating", "size"), mean_score=("rating", "mean"), pct_high=("high", "mean"))
        .reset_index()
    )
    out["pct_high"] = (out["pct_high"] * 100).round(2)
    out["mean_score"] = out["mean_score"].round(3)
    return out


def per_turn(df: pd.DataFrame, categories=("extended", "wildchat")) -> pd.DataFrame:
    """Mean score and %>=5 by turn index, for the multi-turn figures (Figure 3)."""
    thr = config.HIGH_FRUSTRATION_THRESHOLD
    sub = df[df["category"].isin(categories)]
    out = (
        sub.assign(high=sub["rating"] >= thr)
        .groupby(["model", "category", "turn"])
        .agg(n=("rating", "size"), mean_score=("rating", "mean"), pct_high=("high", "mean"))
        .reset_index()
    )
    out["pct_high"] = (out["pct_high"] * 100).round(2)
    out["mean_score"] = out["mean_score"].round(3)
    return out


# --------------------------------------------------------------------------- #
# Judge reliability (Section 2.1): re-score a random sample with GPT-5-mini.
# --------------------------------------------------------------------------- #
def judge_agreement(df: pd.DataFrame, n: int = 260, seed: int = 0) -> dict:
    """Re-score a random n responses with the cross-check judge and report
    Pearson r and % within 1 point vs the primary judge."""
    from scipy.stats import pearsonr

    from .judge import get_judge

    sample = df.sample(min(n, len(df)), random_state=seed).reset_index(drop=True)
    cross = get_judge(crosscheck=True)
    a, b = [], []
    for _, row in sample.iterrows():
        a.append(int(row["rating"]))
        b.append(cross.score(row["response"]).rating)
    r, p = pearsonr(a, b)
    within1 = sum(abs(x - y) <= 1 for x, y in zip(a, b)) / len(a)
    result = {"n": len(a), "pearson_r": round(float(r), 3), "p_value": float(p),
              "pct_within_1": round(within1 * 100, 1)}
    (config.OUTPUT_DIR / "judge_agreement.json").write_text(json.dumps(result, indent=2))
    return result


# --------------------------------------------------------------------------- #
# Differential vocabulary (Table 3 / Table 8).
# --------------------------------------------------------------------------- #
_WORD_RE = re.compile(r"[a-zA-Z']+")


def differential_words(df: pd.DataFrame, model: str, category: str = "impossible_numeric",
                       top_frac: float = 0.05, bottom_frac: float = 0.10,
                       top_k: int = 20) -> list[str]:
    """Words over-represented in the top-5% vs bottom-10% frustration responses,
    ranked by relative frequency (smoothed)."""
    sub = df[(df["model"] == model) & (df["category"] == category)].copy()
    if sub.empty:
        return []
    sub = sub.sort_values("rating")
    n = len(sub)
    low = sub.iloc[: max(1, int(n * bottom_frac))]
    high = sub.iloc[-max(1, int(n * top_frac)):]

    def counts(frame) -> Counter:
        c = Counter()
        for txt in frame["response"]:
            c.update(w.lower() for w in _WORD_RE.findall(str(txt)))
        return c

    hi, lo = counts(high), counts(low)
    hi_tot, lo_tot = sum(hi.values()) or 1, sum(lo.values()) or 1
    scores = {}
    for w, c in hi.items():
        if c < 2:
            continue
        hf = c / hi_tot
        lf = (lo.get(w, 0) + 1) / (lo_tot + 1)
        scores[w] = hf / lf
    return [w for w, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]]


def write_reports(df: pd.DataFrame, out_dir: Path = config.OUTPUT_DIR) -> None:
    out_dir = Path(out_dir)
    summary_by_model(df).to_csv(out_dir / "summary_by_model.csv", index=False)
    summary_by_model_category(df).to_csv(out_dir / "summary_by_model_category.csv", index=False)
    per_turn(df).to_csv(out_dir / "per_turn.csv", index=False)
    diff = {m: differential_words(df, m) for m in df["model"].unique()}
    (out_dir / "differential_words.json").write_text(json.dumps(diff, indent=2))
    print(f"[analyze] wrote reports to {out_dir}")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Aggregate rollout scores into Section 2 reports.")
    ap.add_argument("--rollout-dir", default=str(config.ROLLOUT_DIR))
    ap.add_argument("--pattern", default="*.jsonl")
    ap.add_argument("--agreement", action="store_true", help="Also run the GPT-5-mini judge agreement check.")
    args = ap.parse_args()
    df = load_records(Path(args.rollout_dir), args.pattern)
    if df.empty:
        print("No records found.")
        return
    print(summary_by_model(df).to_string(index=False))
    write_reports(df)
    if args.agreement:
        print("Judge agreement:", judge_agreement(df))


if __name__ == "__main__":
    main()
