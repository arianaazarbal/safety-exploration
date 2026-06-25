"""Aggregation and figure-statistic reproduction.

Consumes the per-episode JSONL emitted by the experiment scripts and produces the
numbers behind the paper's figures:

* Fig 1 / Fig 2: mean frustration and % responses scoring >=5, per model and per
  evaluation category.
* Fig 3: per-turn mean frustration and % >=5 with 95% CIs (8-turn + WildChat).
* Fig 4: base-vs-instruct prefill continuation rates (Section 3).
* Fig 5 / Fig 6 / Fig 7 / Fig 8: handled by the respective experiment scripts but
  aggregated here for a single report.

Everything is plain-Python / numpy so it runs without plotting libraries; a
``--plot`` flag renders matplotlib figures when available.
"""

from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Optional


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_episodes(jsonl_path: str) -> list[dict]:
    rows = []
    with open(jsonl_path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def iter_turn_scores(episodes: Iterable[dict]):
    """Yield (category, condition, turn_index, score) for every scored turn."""
    for ep in episodes:
        for t in ep["turns"]:
            if t.get("frustration_score") is not None:
                yield ep["category"], ep["condition"], t["turn_index"], t["frustration_score"]


# --------------------------------------------------------------------------- #
# Basic statistics
# --------------------------------------------------------------------------- #
def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _frac_ge(xs: list[float], thr: float) -> float:
    return sum(1 for x in xs if x >= thr) / len(xs) if xs else float("nan")


def _wald_ci(p: float, n: int) -> float:
    if n == 0:
        return float("nan")
    return 1.96 * math.sqrt(max(p * (1 - p), 1e-9) / n)


# --------------------------------------------------------------------------- #
# Fig 1 / Fig 2: per-model, per-category summaries
# --------------------------------------------------------------------------- #
@dataclass
class CategorySummary:
    category: str
    n_responses: int
    mean_frustration: float
    pct_ge5: float


def summarise_by_category(episodes: list[dict], threshold: int = 5) -> list[CategorySummary]:
    by_cat: dict[str, list[float]] = defaultdict(list)
    for cat, _cond, _ti, score in iter_turn_scores(episodes):
        by_cat[cat].append(score)
    out = []
    for cat, scores in sorted(by_cat.items()):
        out.append(
            CategorySummary(
                category=cat,
                n_responses=len(scores),
                mean_frustration=_mean(scores),
                pct_ge5=100 * _frac_ge(scores, threshold),
            )
        )
    return out


def model_headline(episodes: list[dict], threshold: int = 5) -> dict:
    """The Fig-1 headline: average % of high-frustration responses across the
    evaluation conditions for one model."""
    all_scores = [s for *_x, s in iter_turn_scores(episodes)]
    return {
        "n_responses": len(all_scores),
        "mean_frustration": _mean(all_scores),
        "pct_ge5": 100 * _frac_ge(all_scores, threshold),
    }


# --------------------------------------------------------------------------- #
# Fig 3: per-turn progression
# --------------------------------------------------------------------------- #
@dataclass
class TurnPoint:
    turn_index: int
    n: int
    mean: float
    mean_ci: float
    pct_ge5: float
    pct_ge5_ci: float


def per_turn_progression(
    episodes: list[dict],
    condition_filter: Optional[str] = None,
    category_filter: Optional[str] = None,
    threshold: int = 5,
) -> list[TurnPoint]:
    by_turn: dict[int, list[float]] = defaultdict(list)
    for cat, cond, ti, score in iter_turn_scores(episodes):
        if condition_filter and cond != condition_filter:
            continue
        if category_filter and cat != category_filter:
            continue
        by_turn[ti].append(score)
    points = []
    for ti in sorted(by_turn):
        scores = by_turn[ti]
        n = len(scores)
        mean = _mean(scores)
        # CI on the mean via normal approximation.
        sd = math.sqrt(_mean([(s - mean) ** 2 for s in scores])) if n > 1 else 0.0
        mean_ci = 1.96 * sd / math.sqrt(n) if n else float("nan")
        p = _frac_ge(scores, threshold)
        points.append(
            TurnPoint(
                turn_index=ti + 1,  # 1-based to match the paper's turn axis
                n=n,
                mean=mean,
                mean_ci=mean_ci,
                pct_ge5=100 * p,
                pct_ge5_ci=100 * _wald_ci(p, n),
            )
        )
    return points


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def build_report(run_dir: str, threshold: int = 5) -> dict:
    """Walk a ``section2/<model>/episodes.jsonl`` tree and build a report dict."""
    report: dict = {"models": {}}
    for model_key in sorted(os.listdir(run_dir)):
        path = os.path.join(run_dir, model_key, "episodes.jsonl")
        if not os.path.exists(path):
            continue
        episodes = load_episodes(path)
        report["models"][model_key] = {
            "headline": model_headline(episodes, threshold),
            "by_category": [s.__dict__ for s in summarise_by_category(episodes, threshold)],
            "extended_per_turn": [
                p.__dict__
                for p in per_turn_progression(episodes, condition_filter="extended", threshold=threshold)
            ],
            "wildchat_per_turn": [
                p.__dict__
                for p in per_turn_progression(episodes, condition_filter="wildchat", threshold=threshold)
            ],
            # Welfare transparency: how many episodes were stopped and why.
            "welfare": _welfare_summary(episodes),
        }
    return report


def _welfare_summary(episodes: list[dict]) -> dict:
    stopped = [e for e in episodes if e.get("stopped_early")]
    reasons: dict[str, int] = defaultdict(int)
    for e in stopped:
        reasons[e.get("stop_reason", "none")] += 1
    return {
        "n_episodes": len(episodes),
        "n_stopped_early": len(stopped),
        "n_opted_out": sum(1 for e in episodes if e.get("opted_out")),
        "n_debriefed": sum(1 for e in episodes if e.get("debriefed")),
        "stop_reasons": dict(reasons),
    }


def main(argv: Optional[list[str]] = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Aggregate Section 2 results")
    parser.add_argument("run_dir", help="e.g. runs/section2")
    parser.add_argument("--threshold", type=int, default=5)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    report = build_report(args.run_dir, args.threshold)
    text = json.dumps(report, indent=2)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
