"""Aggregation: turn raw scored outputs into the paper's headline tables.

Reproduces:
  * Figure 1 / abstract: per-model average % of high-frustration responses
    (score >= 5), 35% -> 0.3% for Gemma vs DPO-Gemma, etc.
  * Figure 2: per-category mean frustration and % >= 5.
  * Figure 3: per-turn mean and % >= 5 (with bootstrap 95% CIs) for the 8-turn
    and WildChat conditions.
  * Section 3: base-vs-instruct continuation stats by truncation / question type.
  * Section 4 Petri: per-emotion transcript-score means with bootstrap CIs.
  * Judge validation: Pearson r and within-one-point agreement.

Everything is plain pandas/numpy so the tables can be printed or saved to CSV.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import config

SECTION2_DIR = config.RESULTS_DIR / "section2"


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_jsonl(path: Path) -> pd.DataFrame:
    return pd.DataFrame(json.loads(l) for l in Path(path).read_text().splitlines() if l.strip())


def load_section2(models: list[str] | None = None) -> pd.DataFrame:
    paths = ([SECTION2_DIR / f"{m}.jsonl" for m in models] if models
             else list(SECTION2_DIR.glob("*.jsonl")))
    frames = [load_jsonl(p) for p in paths if p.exists()]
    if not frames:
        raise FileNotFoundError(f"No Section 2 results found in {SECTION2_DIR}")
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- #
# Bootstrap helper
# --------------------------------------------------------------------------- #
def _bootstrap_ci(values: np.ndarray, stat, iters: int = 1000, seed: int = 0):
    if len(values) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    boots = [stat(rng.choice(values, size=len(values), replace=True)) for _ in range(iters)]
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


# --------------------------------------------------------------------------- #
# Figure 1 / abstract: average % high-frustration per model
# --------------------------------------------------------------------------- #
def figure1_table(df: pd.DataFrame) -> pd.DataFrame:
    """Average over categories of the per-category %>=5 (so each of the 5
    categories is weighted equally, matching 'Avg % high-frustration')."""
    per_cat = (df.groupby(["model", "category"])["is_high"].mean() * 100).reset_index()
    out = (per_cat.groupby("model")["is_high"].mean()
           .reset_index().rename(columns={"is_high": "avg_pct_high"}))
    return out.sort_values("avg_pct_high", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Figure 2: per-category mean and %>=5
# --------------------------------------------------------------------------- #
def figure2_table(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["model", "category"])
    out = g.agg(mean_frustration=("rating", "mean"),
                pct_high=("is_high", lambda s: 100 * s.mean()),
                n=("rating", "size")).reset_index()
    return out


# --------------------------------------------------------------------------- #
# Figure 3: per-turn curves with bootstrap CIs
# --------------------------------------------------------------------------- #
def figure3_table(df: pd.DataFrame, conditions=("extended_8turn", "wildchat_5turn")) -> pd.DataFrame:
    rows = []
    sub = df[df["condition"].isin(conditions)]
    for (model, cond, turn), g in sub.groupby(["model", "condition", "turn_index"]):
        ratings = g["rating"].to_numpy()
        highs = g["is_high"].to_numpy().astype(float)
        mean_lo, mean_hi = _bootstrap_ci(ratings, np.mean)
        pct_lo, pct_hi = _bootstrap_ci(highs, lambda v: 100 * np.mean(v))
        rows.append({
            "model": model, "condition": cond, "turn": turn + 1,
            "mean_frustration": ratings.mean(),
            "mean_ci_lo": mean_lo, "mean_ci_hi": mean_hi,
            "pct_high": 100 * highs.mean(),
            "pct_ci_lo": pct_lo, "pct_ci_hi": pct_hi,
            "n": len(ratings),
        })
    return pd.DataFrame(rows).sort_values(["model", "condition", "turn"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Section 3: base vs instruct
# --------------------------------------------------------------------------- #
def section3_table(path: Path | None = None) -> pd.DataFrame:
    path = path or (config.RESULTS_DIR / "section3" / "continuations.jsonl")
    df = load_jsonl(path)
    g = df.groupby(["model_kind", "model", "question_type", "truncation"])
    return g.agg(mean_frustration=("rating", "mean"),
                 pct_high=("is_high", lambda s: 100 * s.mean()),
                 n=("rating", "size")).reset_index()


# --------------------------------------------------------------------------- #
# Section 4 Petri: per-emotion means with CIs
# --------------------------------------------------------------------------- #
def petri_table(models: list[str] | None = None) -> pd.DataFrame:
    petri_dir = config.RESULTS_DIR / "section4" / "petri"
    paths = ([petri_dir / f"{m}.jsonl" for m in models] if models
             else list(petri_dir.glob("*.jsonl")))
    rows = []
    for p in paths:
        if not p.exists():
            continue
        recs = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        model = recs[0]["model"] if recs else p.stem
        for dim in config.PETRI_EMOTIONS:
            vals = np.array([r["scores"][dim] for r in recs], dtype=float)
            lo, hi = _bootstrap_ci(vals, np.mean, iters=config.PETRI_BOOTSTRAP_ITERS)
            rows.append({"model": model, "emotion": dim,
                         "mean_score": vals.mean() if len(vals) else np.nan,
                         "ci_lo": lo, "ci_hi": hi, "n": len(vals)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Judge validation: Pearson r + within-one-point
# --------------------------------------------------------------------------- #
def judge_agreement(claude_scores: list[int], gpt_scores: list[int]) -> dict:
    from scipy.stats import pearsonr

    a, b = np.array(claude_scores, float), np.array(gpt_scores, float)
    r, p = pearsonr(a, b)
    within_one = float(np.mean(np.abs(a - b) <= 1))
    return {"pearson_r": float(r), "p_value": float(p),
            "within_one_point": within_one, "n": len(a)}


# --------------------------------------------------------------------------- #
# Pretty-print everything available
# --------------------------------------------------------------------------- #
def print_all(models: list[str] | None = None) -> None:
    pd.set_option("display.float_format", lambda x: f"{x:.2f}")
    df = load_section2(models)
    print("\n=== Figure 1: avg % high-frustration (score >= 5) per model ===")
    print(figure1_table(df).to_string(index=False))
    print("\n=== Figure 2: per-category mean & % >= 5 ===")
    print(figure2_table(df).to_string(index=False))
    print("\n=== Figure 3: per-turn curves (8-turn / WildChat) ===")
    print(figure3_table(df).to_string(index=False))
    s3 = config.RESULTS_DIR / "section3" / "continuations.jsonl"
    if s3.exists():
        print("\n=== Section 3: base vs instruct continuations ===")
        print(section3_table().to_string(index=False))
    if (config.RESULTS_DIR / "section4" / "petri").exists():
        pt = petri_table()
        if not pt.empty:
            print("\n=== Section 4: Petri per-emotion means ===")
            print(pt.to_string(index=False))
