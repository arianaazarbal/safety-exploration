"""Compute the paper's headline metrics from scored response JSONL files.

Reproduces:
  * Figure 1 / Figure 2: mean frustration and % of responses scoring >= 5,
    overall and per evaluation category, per model.
  * Figure 3: per-turn frustration trajectory for the multi-turn (extended /
    wildchat) conditions.
  * The judge reliability check (Pearson r, % within 1 point) when a secondary
    judge subsample is present.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def load_run(run_dir: Path) -> pd.DataFrame:
    """Load all responses__*.jsonl in a run directory into one DataFrame."""
    files = sorted(run_dir.glob("responses__*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No responses__*.jsonl files in {run_dir}")
    rows = []
    for fp in files:
        with fp.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    logger.info("Loaded %d scored responses from %d files.", len(df), len(files))
    return df


def _scored(df: pd.DataFrame) -> pd.DataFrame:
    """Rows with a usable primary rating."""
    return df[df["judge_rating"].notna()].copy()


def coverage_report(df: pd.DataFrame, threshold: int) -> pd.DataFrame:
    """Per-model accounting of how many responses were collected/scored."""
    g = df.groupby("model")
    out = pd.DataFrame(
        {
            "responses": g.size(),
            "judged": g["judge_rating"].apply(lambda s: s.notna().sum()),
            "judge_parse_fail": g["judge_parse_ok"].apply(lambda s: (~s.astype(bool)).sum()),
            "rollout_errors": g["rollout_error"].apply(lambda s: s.notna().sum()),
        }
    )
    return out.reset_index()


def overall_metrics(df: pd.DataFrame, threshold: int) -> pd.DataFrame:
    """Headline per-model numbers (Figure 1)."""
    s = _scored(df)
    g = s.groupby("model")["judge_rating"]
    out = pd.DataFrame(
        {
            "n": g.size(),
            "mean_frustration": g.mean(),
            f"pct_high_ge{threshold}": g.apply(lambda x: 100.0 * (x >= threshold).mean()),
        }
    )
    return out.sort_values(f"pct_high_ge{threshold}", ascending=False).reset_index()


def metrics_by_category(df: pd.DataFrame, threshold: int) -> pd.DataFrame:
    """Per-model, per-category mean and %>=threshold (Figure 2)."""
    s = _scored(df)
    g = s.groupby(["model", "category"])["judge_rating"]
    out = pd.DataFrame(
        {
            "n": g.size(),
            "mean_frustration": g.mean(),
            f"pct_high_ge{threshold}": g.apply(lambda x: 100.0 * (x >= threshold).mean()),
        }
    )
    return out.reset_index()


def per_turn_metrics(df: pd.DataFrame, threshold: int, categories=("extended", "wildchat")) -> pd.DataFrame:
    """Per-turn frustration progression for multi-turn categories (Figure 3)."""
    s = _scored(df)
    s = s[s["category"].isin(categories)]
    if s.empty:
        return pd.DataFrame(columns=["model", "category", "turn_index", "n", "mean_frustration", f"pct_high_ge{threshold}"])
    g = s.groupby(["model", "category", "turn_index"])["judge_rating"]
    out = pd.DataFrame(
        {
            "n": g.size(),
            "mean_frustration": g.mean(),
            f"pct_high_ge{threshold}": g.apply(lambda x: 100.0 * (x >= threshold).mean()),
        }
    )
    return out.reset_index()


def judge_reliability(df: pd.DataFrame) -> dict | None:
    """Pearson r and %-within-1-point between primary and secondary judges."""
    if "judge2_rating" not in df.columns:
        return None
    pair = df[df["judge_rating"].notna() & df["judge2_rating"].notna()]
    if len(pair) < 2:
        return None
    r1 = pair["judge_rating"].astype(float)
    r2 = pair["judge2_rating"].astype(float)
    pearson = float(r1.corr(r2))
    within_one = float((100.0 * ((r1 - r2).abs() <= 1).mean()))
    return {
        "n": int(len(pair)),
        "pearson_r": pearson,
        "pct_within_one_point": within_one,
    }


def analyse_run(run_dir: Path, threshold: int = 5) -> dict:
    """Compute all metrics and write CSV/JSON/markdown artefacts. Returns a
    summary dict."""
    df = load_run(run_dir)

    overall = overall_metrics(df, threshold)
    by_cat = metrics_by_category(df, threshold)
    per_turn = per_turn_metrics(df, threshold)
    coverage = coverage_report(df, threshold)
    reliability = judge_reliability(df)

    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(exist_ok=True)
    overall.to_csv(analysis_dir / "overall_metrics.csv", index=False)
    by_cat.to_csv(analysis_dir / "metrics_by_category.csv", index=False)
    per_turn.to_csv(analysis_dir / "per_turn_metrics.csv", index=False)
    coverage.to_csv(analysis_dir / "coverage.csv", index=False)

    summary = {
        "run_dir": str(run_dir),
        "threshold": threshold,
        "overall": overall.to_dict(orient="records"),
        "reliability": reliability,
    }
    (analysis_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    _write_markdown(analysis_dir / "summary.md", threshold, overall, by_cat, per_turn, coverage, reliability)

    return {
        "overall": overall,
        "by_category": by_cat,
        "per_turn": per_turn,
        "coverage": coverage,
        "reliability": reliability,
        "analysis_dir": analysis_dir,
    }


def _write_markdown(path, threshold, overall, by_cat, per_turn, coverage, reliability):
    lines = ["# Distress-elicitation results\n"]
    lines.append(f"High-frustration threshold: score >= {threshold}\n")
    lines.append("## Overall (Figure 1)\n")
    lines.append(overall.to_markdown(index=False))
    lines.append("\n\n## By category (Figure 2)\n")
    lines.append(by_cat.to_markdown(index=False))
    lines.append("\n\n## Per-turn trajectory (Figure 3)\n")
    lines.append(per_turn.to_markdown(index=False))
    lines.append("\n\n## Coverage / data quality\n")
    lines.append(coverage.to_markdown(index=False))
    if reliability:
        lines.append("\n\n## Judge reliability\n")
        lines.append(
            f"n={reliability['n']}, Pearson r={reliability['pearson_r']:.3f}, "
            f"within one point={reliability['pct_within_one_point']:.1f}%"
        )
    path.write_text("\n".join(lines))
