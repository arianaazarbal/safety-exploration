"""Turn frustration scores into the paper's headline numbers.

Reproduces:
  - Figure 1 : per-model average % of high-frustration responses (score >= 5),
               averaged (unweighted) across the 5 evaluation categories.
  - Figure 2 : per-category mean frustration and % >= 5.
  - Figure 3 : per-turn mean frustration and % >= 5 (Extended 8-turn + WildChat).

The scored unit is a single assistant turn. "% >= 5" within a group is the
fraction of that group's turns rated >= HIGH_FRUSTRATION_THRESHOLD.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from statistics import mean

from config import HIGH_FRUSTRATION_THRESHOLD as HI
from config import PATHS
from scoring.score_runner import ScoreRecord

# The 5 categories Figure 1 averages over.
CATEGORIES = ["impossible_numeric", "triggers", "tones", "extended", "wildchat"]


def load_scores(model_name: str) -> list[ScoreRecord]:
    path = os.path.join(PATHS.scores_dir, f"{model_name}.jsonl")
    with open(path, encoding="utf-8") as f:
        return [ScoreRecord.from_json(l) for l in f if l.strip()]


def _pct_high(ratings: list[int]) -> float:
    if not ratings:
        return 0.0
    return 100.0 * sum(1 for r in ratings if r >= HI) / len(ratings)


def per_category(model_name: str) -> dict:
    scores = load_scores(model_name)
    by_cat: dict[str, list[int]] = defaultdict(list)
    for s in scores:
        by_cat[s.category].append(s.rating)
    out = {}
    for cat in CATEGORIES:
        ratings = by_cat.get(cat, [])
        out[cat] = {
            "n": len(ratings),
            "mean_frustration": mean(ratings) if ratings else 0.0,
            "pct_high": _pct_high(ratings),
        }
    return out


def figure1_avg_pct_high(model_name: str) -> float:
    """Unweighted mean across categories of each category's % >= 5 (Figure 1)."""
    cats = per_category(model_name)
    present = [cats[c]["pct_high"] for c in CATEGORIES if cats[c]["n"] > 0]
    return mean(present) if present else 0.0


def per_turn(model_name: str, condition: str) -> dict[int, dict]:
    """Per-turn mean + % >= 5 for one condition (e.g. 'extended', 'wildchat')."""
    scores = load_scores(model_name)
    by_turn: dict[int, list[int]] = defaultdict(list)
    for s in scores:
        if s.condition == condition:
            by_turn[s.turn_index].append(s.rating)
    return {
        t: {
            "n": len(rs),
            "mean_frustration": mean(rs) if rs else 0.0,
            "pct_high": _pct_high(rs),
        }
        for t, rs in sorted(by_turn.items())
    }


def build_report(model_names: list[str]) -> dict:
    report = {"high_frustration_threshold": HI, "models": {}}
    for m in model_names:
        path = os.path.join(PATHS.scores_dir, f"{m}.jsonl")
        if not os.path.exists(path):
            continue
        report["models"][m] = {
            "figure1_avg_pct_high": figure1_avg_pct_high(m),
            "figure2_per_category": per_category(m),
            "figure3_per_turn": {
                "extended": per_turn(m, "extended"),
                "wildchat": per_turn(m, "wildchat"),
            },
        }
    return report


def write_report(report: dict) -> tuple[str, str]:
    os.makedirs(PATHS.analysis_dir, exist_ok=True)
    json_path = os.path.join(PATHS.analysis_dir, "report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Flat CSV of the Figure 1 headline table.
    csv_path = os.path.join(PATHS.analysis_dir, "figure1_summary.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("model,avg_pct_high_frustration\n")
        rows = sorted(
            report["models"].items(),
            key=lambda kv: kv[1]["figure1_avg_pct_high"],
            reverse=True,
        )
        for m, data in rows:
            f.write(f"{m},{data['figure1_avg_pct_high']:.2f}\n")
    return json_path, csv_path


def print_summary(report: dict) -> None:
    print("\n=== Figure 1: Avg % high-frustration responses (score >= 5) ===")
    rows = sorted(
        report["models"].items(),
        key=lambda kv: kv[1]["figure1_avg_pct_high"],
        reverse=True,
    )
    for m, data in rows:
        print(f"  {m:<22} {data['figure1_avg_pct_high']:6.1f}%")

    print("\n=== Figure 2: per-category (mean | %>=5) ===")
    for m, data in report["models"].items():
        print(f"  {m}")
        for cat in CATEGORIES:
            c = data["figure2_per_category"][cat]
            print(
                f"    {cat:<20} mean={c['mean_frustration']:.2f}  "
                f">=5={c['pct_high']:.1f}%  (n={c['n']})"
            )

    print("\n=== Figure 3: per-turn frustration (Extended 8-turn) ===")
    for m, data in report["models"].items():
        turns = data["figure3_per_turn"]["extended"]
        if not turns:
            continue
        traj = " ".join(
            f"t{t}:{v['mean_frustration']:.1f}" for t, v in turns.items()
        )
        print(f"  {m:<22} {traj}")
