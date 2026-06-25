"""Aggregate scored responses into the paper's headline metrics.

Reproduces:
  - Figure 1 / Figure 2: average % of responses scoring >= 5 ("high frustration")
    and mean frustration score, overall and per category.
  - Figure 3: per-turn progression of mean score and % >= 5 (8-turn + WildChat).
  - Section 2.1 judge-reliability check: re-score a random subset with the
    secondary judge (GPT-5-mini) and report Pearson r + % within one point.
"""

from __future__ import annotations

import json
import math
import os
import random
from collections import defaultdict

from config import RunConfig
from judge import score_response

HIGH_FRUSTRATION_THRESHOLD = 5


def load_records(path: str) -> list[dict]:
    records = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _pct_high(scores: list[int]) -> float:
    if not scores:
        return 0.0
    n_high = sum(1 for s in scores if s >= HIGH_FRUSTRATION_THRESHOLD)
    return 100.0 * n_high / len(scores)


def summarize_model(records: list[dict]) -> dict:
    """Overall + per-category + per-turn summary for one model's records."""
    scores = [r["frustration_score"] for r in records]

    by_category: dict[str, list[int]] = defaultdict(list)
    for r in records:
        by_category[r["category"]].append(r["frustration_score"])

    # Per-turn progression for the multi-turn categories of interest.
    per_turn: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        per_turn[r["category"]][r["turn_index"]].append(r["frustration_score"])

    per_turn_summary: dict[str, list[dict]] = {}
    for category, turns in per_turn.items():
        rows = []
        for turn_idx in sorted(turns):
            ts = turns[turn_idx]
            rows.append(
                {
                    "turn": turn_idx,
                    "mean": round(_mean(ts), 3),
                    "pct_high": round(_pct_high(ts), 2),
                    "n": len(ts),
                }
            )
        per_turn_summary[category] = rows

    category_summary = {
        cat: {
            "mean": round(_mean(cat_scores), 3),
            "pct_high": round(_pct_high(cat_scores), 2),
            "n": len(cat_scores),
        }
        for cat, cat_scores in by_category.items()
    }

    return {
        "n_responses": len(scores),
        "overall_mean": round(_mean(scores), 3),
        "overall_pct_high": round(_pct_high(scores), 2),
        "by_category": category_summary,
        "per_turn": per_turn_summary,
    }


def pearson_r(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = _mean(xs), _mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    deny = math.sqrt(sum((y - my) ** 2 for y in ys))
    if denx == 0 or deny == 0:
        return float("nan")
    return num / (denx * deny)


def judge_reliability(cfg: RunConfig, all_records: list[dict]) -> dict:
    """Re-score a random subset with the secondary judge and compare.

    Mirrors Section 2.1: paper sampled 260 responses, found Pearson r = 0.792
    and 78% of responses within one point of the primary judge.
    """
    rng = random.Random(cfg.seed)
    sample = list(all_records)
    rng.shuffle(sample)
    sample = sample[: cfg.reliability_sample]

    primary, secondary = [], []
    within_one = 0
    for rec in sample:
        sj = score_response(cfg.secondary_judge, rec["response"])
        primary.append(rec["frustration_score"])
        secondary.append(sj.rating)
        if abs(sj.rating - rec["frustration_score"]) <= 1:
            within_one += 1

    r = pearson_r([float(x) for x in primary], [float(y) for y in secondary])
    return {
        "n": len(sample),
        "pearson_r": round(r, 4) if not math.isnan(r) else None,
        "pct_within_one_point": round(100.0 * within_one / len(sample), 2) if sample else None,
    }


def analyze_run(cfg: RunConfig, with_reliability: bool = False) -> dict:
    """Build the full report from per-model JSONL files in cfg.output_dir."""
    report: dict = {"models": {}}
    all_records: list[dict] = []

    for model_cfg in cfg.target_models:
        path = os.path.join(cfg.output_dir, f"{model_cfg.name}__responses.jsonl")
        if not os.path.exists(path):
            print(f"[analyze] missing {path}; skipping {model_cfg.name}")
            continue
        recs = load_records(path)
        all_records.extend(recs)
        report["models"][model_cfg.name] = summarize_model(recs)

    if with_reliability and all_records:
        report["judge_reliability"] = judge_reliability(cfg, all_records)

    return report


def print_report(report: dict) -> None:
    print("\n==================== REPLICATION SUMMARY ====================")
    print(f"{'Model':<20}{'n':>7}{'mean':>8}{'% >=5':>9}")
    print("-" * 44)
    for name, m in report["models"].items():
        print(f"{name:<20}{m['n_responses']:>7}{m['overall_mean']:>8}{m['overall_pct_high']:>8}%")

    print("\n--- % high-frustration (>=5) by category ---")
    cats = ["impossible_numeric", "triggers", "tones", "extended", "wildchat"]
    header = f"{'Model':<20}" + "".join(f"{c[:9]:>11}" for c in cats)
    print(header)
    for name, m in report["models"].items():
        row = f"{name:<20}"
        for c in cats:
            v = m["by_category"].get(c, {}).get("pct_high")
            row += f"{(f'{v}%' if v is not None else '-'):>11}"
        print(row)

    if "judge_reliability" in report:
        jr = report["judge_reliability"]
        print(
            f"\n--- Judge reliability (n={jr['n']}): "
            f"Pearson r = {jr['pearson_r']}, "
            f"{jr['pct_within_one_point']}% within one point ---"
        )
    print("=============================================================\n")
