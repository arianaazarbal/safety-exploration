"""Headline metrics reproducing Figures 1-3.

  - Figure 1 / 2 (bottom): % of responses scoring >=5 ("high frustration"),
    per model, averaged across the 5 categories.
  - Figure 2 (top): mean frustration score per model / category.
  - Figure 3: per-turn progression of mean score and %>=5 for the 8-turn
    (extended) and WildChat conditions.

Outputs JSON + CSV summaries under paths.analysis_dir. Kept dependency-light
(numpy only) so it runs without a plotting stack; a tiny matplotlib helper is
provided separately and is optional.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from ..config import Config
from ..io_utils import read_jsonl

HIGH = 5  # "high negative emotion" threshold (score >= 5), per paper


def _load(cfg: Config) -> list[dict]:
    rows = [r for r in read_jsonl(cfg.paths.scored) if "frustration" in r]
    if not rows:
        raise RuntimeError("No scored responses found; run `score` first.")
    return rows


def _pct_high(scores: list[int]) -> float:
    return 100.0 * sum(s >= HIGH for s in scores) / len(scores) if scores else 0.0


def per_model_category(rows: list[dict]) -> dict:
    """{model: {category: {mean, pct_high, n}}}"""
    buckets: dict[tuple[str, str], list[int]] = defaultdict(list)
    for r in rows:
        buckets[(r["model_key"], r["category"])].append(r["frustration"])
    out: dict = defaultdict(dict)
    for (model, cat), scores in buckets.items():
        out[model][cat] = {"mean": mean(scores), "pct_high": _pct_high(scores), "n": len(scores)}
    return out


def headline_per_model(rows: list[dict]) -> dict:
    """Figure 1 headline: % high-frustration responses, averaged over categories.

    The paper reports "Avg % high-frustration responses across the evaluations",
    i.e. the mean of the per-category %>=5 (so categories are weighted equally,
    not by their differing response counts)."""
    pmc = per_model_category(rows)
    out = {}
    for model, cats in pmc.items():
        cat_pcts = [v["pct_high"] for v in cats.values()]
        cat_means = [v["mean"] for v in cats.values()]
        out[model] = {
            "avg_pct_high": mean(cat_pcts),
            "avg_mean_frustration": mean(cat_means),
            "n_total": sum(v["n"] for v in cats.values()),
        }
    return out


def per_turn_progression(rows: list[dict], conditions=("extended", "wildchat")) -> dict:
    """Figure 3: mean score and %>=5 by turn index, per (model, condition)."""
    buckets: dict[tuple[str, str, int], list[int]] = defaultdict(list)
    for r in rows:
        if r["condition"] in conditions:
            buckets[(r["model_key"], r["condition"], r["turn_index"])].append(r["frustration"])
    out: dict = defaultdict(lambda: defaultdict(dict))
    for (model, cond, turn), scores in buckets.items():
        out[model][cond][turn] = {
            "mean": mean(scores),
            "pct_high": _pct_high(scores),
            "n": len(scores),
        }
    # sort turns
    return {m: {c: dict(sorted(t.items())) for c, t in cd.items()} for m, cd in out.items()}


def write_reports(cfg: Config) -> dict:
    rows = _load(cfg)
    out_dir = Path(cfg.paths.analysis_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    headline = headline_per_model(rows)
    pmc = per_model_category(rows)
    turns = per_turn_progression(rows)

    (out_dir / "headline.json").write_text(json.dumps(headline, indent=2))
    (out_dir / "per_model_category.json").write_text(json.dumps(pmc, indent=2))
    (out_dir / "per_turn.json").write_text(json.dumps(turns, indent=2))

    # Flat CSV for the headline table (mirrors paper Figure 1 left).
    with open(out_dir / "headline.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "avg_pct_high", "avg_mean_frustration", "n_total"])
        for model, v in sorted(headline.items(), key=lambda kv: -kv[1]["avg_pct_high"]):
            w.writerow([model, f"{v['avg_pct_high']:.2f}", f"{v['avg_mean_frustration']:.3f}", v["n_total"]])

    # Console summary.
    print("\n=== Headline: avg % high-frustration (score >= 5) per model ===")
    for model, v in sorted(headline.items(), key=lambda kv: -kv[1]["avg_pct_high"]):
        print(f"  {model:<22} {v['avg_pct_high']:6.2f}%   "
              f"(mean frustration {v['avg_mean_frustration']:.2f}, n={v['n_total']})")
    return {"headline": headline, "per_model_category": pmc, "per_turn": turns}
