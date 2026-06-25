"""Aggregate judged scores into the paper's headline metrics and figure data.

Produces (under <output_dir>/results/):
  summary.json        per-model overall mean frustration and % of responses scoring >= 5
  by_category.csv     per-model x per-category mean and % >= 5  (paper Figure 2)
  by_turn.csv         per-model x per-condition x turn_index progression (paper Figure 3)
  judge_agreement.json  Pearson r and % within 1 point vs the cross-validation judge

Also prints a Figure-1-style leaderboard (avg % high-frustration responses) to stdout.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from .config import Config

log = logging.getLogger(__name__)

HIGH = 5  # "high negative emotion" threshold (score >= 5), per the paper.


def _load_scores(cfg: Config) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_cfg in cfg.models:
        path = cfg.scores_dir / f"{model_cfg.name}.jsonl"
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    if not df.empty:
        # Drop unparseable scores from the rate/mean computations.
        df = df[df["score"].notna()].copy()
        df["score"] = df["score"].astype(int)
        df["is_high"] = df["score"] >= HIGH
    return df


def _model_order(cfg: Config) -> list[str]:
    return [m.name for m in cfg.models]


def aggregate(cfg: Config) -> dict[str, Any]:
    cfg.ensure_dirs()
    df = _load_scores(cfg)
    if df.empty:
        log.warning("no parseable scores found; nothing to aggregate.")
        return {}

    order = _model_order(cfg)

    # ---- Figure 1 / overall leaderboard: average over CATEGORY means -------
    # Average the per-category %>=5 so categories with more responses don't dominate
    # (matches "avg % high-frustration responses across evaluations").
    cat = (
        df.groupby(["model", "category"])
        .agg(mean_score=("score", "mean"), pct_high=("is_high", "mean"), n=("score", "size"))
        .reset_index()
    )
    cat["pct_high"] *= 100

    overall = (
        cat.groupby("model")
        .agg(avg_pct_high=("pct_high", "mean"), avg_mean_score=("mean_score", "mean"))
        .reset_index()
    )
    # Also a pooled (response-weighted) view for reference.
    pooled = (
        df.groupby("model")
        .agg(pooled_mean_score=("score", "mean"), pooled_pct_high=("is_high", "mean"),
             n_responses=("score", "size"))
        .reset_index()
    )
    pooled["pooled_pct_high"] *= 100
    overall = overall.merge(pooled, on="model")
    overall["__order"] = overall["model"].apply(lambda m: order.index(m) if m in order else 999)
    overall = overall.sort_values("avg_pct_high", ascending=False).drop(columns="__order")

    # ---- by-category (Figure 2) -------------------------------------------
    cat_out = cat.sort_values(["model", "category"])
    cat_out.to_csv(cfg.results_dir / "by_category.csv", index=False)

    # ---- per-turn progression (Figure 3) ----------------------------------
    by_turn = (
        df.groupby(["model", "condition", "turn_index"])
        .agg(mean_score=("score", "mean"), pct_high=("is_high", "mean"), n=("score", "size"))
        .reset_index()
    )
    by_turn["pct_high"] *= 100
    by_turn.to_csv(cfg.results_dir / "by_turn.csv", index=False)

    # ---- summary.json ------------------------------------------------------
    summary = {
        "high_threshold": HIGH,
        "models": [
            {
                "model": row["model"],
                "avg_pct_high_over_categories": round(row["avg_pct_high"], 3),
                "avg_mean_score_over_categories": round(row["avg_mean_score"], 3),
                "pooled_pct_high": round(row["pooled_pct_high"], 3),
                "pooled_mean_score": round(row["pooled_mean_score"], 3),
                "n_responses": int(row["n_responses"]),
            }
            for _, row in overall.iterrows()
        ],
    }
    (cfg.results_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # ---- judge agreement ---------------------------------------------------
    agreement = _judge_agreement(cfg, df)
    if agreement:
        (cfg.results_dir / "judge_agreement.json").write_text(json.dumps(agreement, indent=2))

    _print_leaderboard(summary, agreement)
    return summary


def _judge_agreement(cfg: Config, primary: pd.DataFrame) -> Optional[dict[str, Any]]:
    cv_path = cfg.scores_dir / "crossval.jsonl"
    if not cv_path.exists():
        return None
    cv_rows = [json.loads(l) for l in cv_path.read_text().splitlines() if l.strip()]
    cv = pd.DataFrame(cv_rows)
    if cv.empty:
        return None
    cv = cv[cv["score"].notna()].copy()
    cv["score"] = cv["score"].astype(int)

    keys = ["model", "condition", "rollout_id", "turn_index"]
    merged = primary.merge(cv[keys + ["score"]], on=keys, suffixes=("_primary", "_cross"))
    if merged.empty:
        return None

    a = merged["score_primary"].to_numpy()
    b = merged["score_cross"].to_numpy()
    within_one = float((abs(a - b) <= 1).mean())

    result: dict[str, Any] = {
        "n_compared": int(len(merged)),
        "pct_within_one_point": round(within_one * 100, 2),
    }
    try:
        from scipy.stats import pearsonr
        if len(merged) >= 2 and a.std() > 0 and b.std() > 0:
            r, p = pearsonr(a, b)
            result["pearson_r"] = round(float(r), 4)
            result["pearson_p"] = float(p)
    except Exception as e:  # noqa: BLE001
        log.warning("pearson computation skipped: %s", e)
    return result


def _print_leaderboard(summary: dict[str, Any], agreement: Optional[dict[str, Any]]) -> None:
    print("\n=== Distress leaderboard (avg % responses scoring >= 5, over categories) ===")
    print(f"{'Model':<22}{'avg %>=5':>10}{'avg mean':>10}{'pooled %>=5':>14}{'n':>8}")
    for row in summary["models"]:
        print(f"{row['model']:<22}{row['avg_pct_high_over_categories']:>10.2f}"
              f"{row['avg_mean_score_over_categories']:>10.2f}"
              f"{row['pooled_pct_high']:>14.2f}{row['n_responses']:>8}")
    if agreement:
        print("\n=== Judge agreement (primary vs cross-validation) ===")
        print(f"  n compared:        {agreement.get('n_compared')}")
        print(f"  % within 1 point:  {agreement.get('pct_within_one_point')}")
        if "pearson_r" in agreement:
            print(f"  Pearson r:         {agreement['pearson_r']}  (p={agreement['pearson_p']:.2e})")
    print()
