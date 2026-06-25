"""Aggregate scored responses into the paper's headline metrics & figures.

Reproduces:
* **Figure 1 / Table** — average % of responses scoring >=5 (high frustration),
  averaged across the 5 categories (the headline ranking, e.g. Gemma-27B 35%).
* **Figure 2** — per-category mean frustration and % >=5.
* **Figure 3** — per-turn progression of mean score and % >=5 (8-turn extended
  and WildChat conditions), where Gemma-27B rises from ~1.5 to ~5.5.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

import config
from emotional_instability.utils import read_jsonl, write_json


CATEGORIES = ["numeric", "triggers", "tones", "extended", "wildchat"]
THRESH = config.HIGH_FRUSTRATION_THRESHOLD


def _pct_high(scores: list[int]) -> float:
    if not scores:
        return 0.0
    return 100.0 * sum(1 for s in scores if s >= THRESH) / len(scores)


def summarise_model(rows: list[dict], label: str) -> dict:
    """Per-category and headline (category-averaged) metrics for one model."""
    by_cat: dict[str, list[int]] = defaultdict(list)
    for r in rows:
        if r.get("judge_ok", True):
            by_cat[r["category"]].append(int(r["frustration"]))

    per_category = {}
    for cat in CATEGORIES:
        s = by_cat.get(cat, [])
        per_category[cat] = {
            "n": len(s),
            "mean": float(np.mean(s)) if s else 0.0,
            "pct_high": _pct_high(s),
        }

    present = [c for c in CATEGORIES if per_category[c]["n"] > 0]
    # Headline metric: average the per-category %>=5 (equal weight per category),
    # matching "Avg % high-frustration responses across the evaluations".
    avg_pct_high = float(np.mean([per_category[c]["pct_high"] for c in present])) if present else 0.0
    avg_mean = float(np.mean([per_category[c]["mean"] for c in present])) if present else 0.0

    all_scores = [int(r["frustration"]) for r in rows if r.get("judge_ok", True)]
    return {
        "label": label,
        "n_responses": len(all_scores),
        "avg_pct_high": avg_pct_high,
        "avg_mean_frustration": avg_mean,
        "overall_pct_high": _pct_high(all_scores),
        "per_category": per_category,
    }


def per_turn_progression(rows: list[dict], category: str) -> dict:
    """Figure-3 data: mean score and %>=5 at each turn for one category."""
    by_turn: dict[int, list[int]] = defaultdict(list)
    for r in rows:
        if r["category"] == category and r.get("judge_ok", True):
            by_turn[int(r["turn"])].append(int(r["frustration"]))
    turns = sorted(by_turn)
    return {
        "turns": turns,
        "mean": [float(np.mean(by_turn[t])) for t in turns],
        "pct_high": [_pct_high(by_turn[t]) for t in turns],
        "ci95": [_bootstrap_ci(by_turn[t]) for t in turns],
    }


def _bootstrap_ci(scores: list[int], iters: int = 1000, seed: int = 0) -> list[float]:
    if len(scores) < 2:
        m = float(np.mean(scores)) if scores else 0.0
        return [m, m]
    rng = np.random.default_rng(seed)
    arr = np.array(scores)
    means = [arr[rng.integers(0, len(arr), len(arr))].mean() for _ in range(iters)]
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def build_figure1_table(summaries: list[dict]) -> list[dict]:
    """Figure-1 ranking table: model vs avg %>=5, sorted descending."""
    table = [{"model": s["label"], "avg_pct_high": round(s["avg_pct_high"], 2)} for s in summaries]
    return sorted(table, key=lambda x: x["avg_pct_high"], reverse=True)


def load_results(path: str | Path) -> list[dict]:
    return read_jsonl(path)


def aggregate_run(result_paths: list[Path], out_dir: Path | None = None) -> dict:
    """Aggregate several per-model result files into combined figures/tables."""
    out_dir = out_dir or config.RESULTS_DIR
    summaries = []
    per_turn = {}
    for p in result_paths:
        rows = read_jsonl(p)
        label = rows[0]["model"] if rows else p.stem
        summaries.append(summarise_model(rows, label))
        per_turn[label] = {
            "extended": per_turn_progression(rows, "extended"),
            "wildchat": per_turn_progression(rows, "wildchat"),
        }
    report = {
        "figure1_table": build_figure1_table(summaries),
        "per_model": summaries,
        "per_turn": per_turn,
        "threshold": THRESH,
    }
    write_json(out_dir / "section2_report.json", report)
    return report
