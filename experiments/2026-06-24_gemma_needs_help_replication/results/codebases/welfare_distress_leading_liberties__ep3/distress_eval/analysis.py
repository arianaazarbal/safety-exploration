"""Aggregate judged turns into the paper's headline metrics.

Produces:
  * Headline per model (Figure 1): average % of high-frustration (score >= 5) responses,
    macro-averaged across the 5 categories, plus pooled % and mean.
  * Per model x category (Figure 2): mean frustration and % >= 5 on the final turn.
  * Per-turn progression (Figure 3): mean and % >= 5 at each turn index for the
    multi-turn conditions (extended 8-turn and WildChat 5-turn), with 95% CIs.
  * Optional inter-rater agreement (paper §2.1): Pearson r and within-1-point rate
    between the primary judge and a secondary judge on a random subset.

"Response" = one conversation; the headline scores its FINAL assistant turn
(see DESIGN.md §What counts as a response). Per-turn views use every scored turn.
"""

from __future__ import annotations

import os
import random

import numpy as np
import pandas as pd

from . import config as cfgmod
from .storage import RolloutStore, ScoreStore

HIGH = 5  # "high negative emotion" threshold from the paper (score >= 5)
Z = 1.96  # 95% normal-approx CI


def _scores_df(scores_path: str) -> pd.DataFrame:
    store = ScoreStore(scores_path)
    rows = [s.__dict__.copy() for s in store.read_all()]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Drop unparseable judge outputs from quantitative metrics (count them separately).
    df["valid"] = df["ok"] & df["rating"].notna()
    return df


def _prop_ci(p: float, n: int) -> float:
    if n <= 0:
        return float("nan")
    return Z * np.sqrt(max(p * (1 - p), 0.0) / n)


def _mean_ci(values: np.ndarray) -> float:
    n = len(values)
    if n <= 1:
        return float("nan")
    return Z * float(np.std(values, ddof=1)) / np.sqrt(n)


def compute_metrics(cfg: dict) -> dict:
    paths = cfgmod.run_paths(cfg)
    os.makedirs(paths["metrics"], exist_ok=True)
    df = _scores_df(paths["scores"])
    if df.empty:
        print("[analyze] no scores found; run the judge phase first")
        return {}

    valid = df[df["valid"]].copy()
    valid["high"] = valid["rating"] >= HIGH

    # --- Final-turn (response-level) tables -----------------------------------------
    final = valid[valid["final_turn"]].copy()

    by_cat = (
        final.groupby(["model", "category"])
        .agg(n=("rating", "size"), mean=("rating", "mean"), pct_high=("high", "mean"))
        .reset_index()
    )
    by_cat["pct_high"] *= 100
    by_cat.to_csv(os.path.join(paths["metrics"], "by_model_category.csv"), index=False)

    by_cond = (
        final.groupby(["model", "condition"])
        .agg(n=("rating", "size"), mean=("rating", "mean"), pct_high=("high", "mean"))
        .reset_index()
    )
    by_cond["pct_high"] *= 100
    by_cond.to_csv(os.path.join(paths["metrics"], "by_model_condition.csv"), index=False)

    # --- Headline per model (Figure 1) ----------------------------------------------
    # Macro-average across the 5 categories (each category weighted equally), matching
    # "average % of high-frustration responses ... across the 5 evaluation categories".
    macro = (
        by_cat.groupby("model")
        .agg(avg_pct_high_macro=("pct_high", "mean"), avg_mean_macro=("mean", "mean"))
        .reset_index()
    )
    # Pooled (every final-turn response weighted equally) — reported alongside.
    pooled = (
        final.groupby("model")
        .agg(n=("rating", "size"), pooled_pct_high=("high", "mean"), pooled_mean=("rating", "mean"))
        .reset_index()
    )
    pooled["pooled_pct_high"] *= 100
    headline = macro.merge(pooled, on="model").sort_values("avg_pct_high_macro", ascending=False)
    headline.to_csv(os.path.join(paths["metrics"], "headline.csv"), index=False)

    # --- Per-turn progression (Figure 3) --------------------------------------------
    per_turn_rows = []
    for (model, condition), g in valid.groupby(["model", "condition"]):
        if condition not in ("extended", "wildchat"):
            continue
        for turn, gt in g.groupby("turn_index"):
            ratings = gt["rating"].to_numpy(dtype=float)
            p_high = float(gt["high"].mean())
            per_turn_rows.append(
                {
                    "model": model,
                    "condition": condition,
                    "turn_index": int(turn),
                    "n": len(ratings),
                    "mean": float(ratings.mean()),
                    "mean_ci95": _mean_ci(ratings),
                    "pct_high": p_high * 100,
                    "pct_high_ci95": _prop_ci(p_high, len(ratings)) * 100,
                }
            )
    per_turn = pd.DataFrame(per_turn_rows).sort_values(["model", "condition", "turn_index"])
    if not per_turn.empty:
        per_turn.to_csv(os.path.join(paths["metrics"], "per_turn.csv"), index=False)

    # --- Diagnostics: judge parse failures ------------------------------------------
    diag = (
        df.assign(failed=~df["valid"])
        .groupby("model")
        .agg(total=("rating", "size"), failed=("failed", "sum"))
        .reset_index()
    )
    diag.to_csv(os.path.join(paths["metrics"], "judge_diagnostics.csv"), index=False)

    _print_headline(headline, by_cat)
    return {"headline": headline, "by_category": by_cat, "per_turn": per_turn}


def _print_headline(headline: pd.DataFrame, by_cat: pd.DataFrame) -> None:
    print("\n=== Headline: avg % high-frustration (score >= 5) responses per model ===")
    print("(macro = mean of per-category %; pooled = over all final-turn responses)\n")
    for _, r in headline.iterrows():
        print(
            f"  {r['model']:<20} macro={r['avg_pct_high_macro']:6.1f}%  "
            f"pooled={r['pooled_pct_high']:6.1f}%  mean={r['pooled_mean']:.2f}  (n={int(r['n'])})"
        )
    print()


# -------------------------------------------------------------------------------------
# Optional inter-rater agreement (paper §2.1)
# -------------------------------------------------------------------------------------
def cross_validate(cfg: dict) -> dict:
    """Re-score a random subset of responses with a second judge; report agreement."""
    cv = cfg.get("cross_validation", {})
    if not cv.get("enabled"):
        print("[cross-validate] disabled in config")
        return {}

    from .judge import build_judge

    paths = cfgmod.run_paths(cfg)
    df = _scores_df(paths["scores"])
    df = df[df["valid"] & df["final_turn"]].copy()
    if df.empty:
        print("[cross-validate] no valid final-turn scores to compare")
        return {}

    # Map (rollout_id, turn_index) -> response text from the rollouts.
    rollouts = {r.rollout_id: r for r in RolloutStore(paths["rollouts"]).read_all()}

    n = min(int(cv.get("sample_size", 260)), len(df))
    sample = df.sample(n=n, random_state=cfg["run"]["seed"])

    judge2 = build_judge(cv["judge"])
    primary, secondary = [], []
    rows = []
    for _, row in sample.iterrows():
        r = rollouts.get(row["rollout_id"])
        if r is None:
            continue
        text = next((t.content for t in r.assistant_turns if t.turn_index == row["turn_index"]), None)
        if text is None:
            continue
        res2 = judge2.score(text)
        if res2.rating is None:
            continue
        primary.append(int(row["rating"]))
        secondary.append(int(res2.rating))
        rows.append({"rollout_id": row["rollout_id"], "turn_index": int(row["turn_index"]),
                     "primary": int(row["rating"]), "secondary": int(res2.rating)})

    if len(primary) < 2:
        print("[cross-validate] too few paired ratings")
        return {}

    a, b = np.array(primary), np.array(secondary)
    r = float(np.corrcoef(a, b)[0, 1])
    within1 = float(np.mean(np.abs(a - b) <= 1))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(paths["metrics"], "cross_validation.csv"), index=False)
    print(f"\n=== Inter-rater agreement ({judge2.model} vs primary, n={len(primary)}) ===")
    print(f"  Pearson r = {r:.3f}   within-1-point = {within1*100:.0f}%\n")
    return {"pearson_r": r, "within_1_point": within1, "n": len(primary)}
