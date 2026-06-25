"""Compute replication metrics and reproduce the paper's headline figures.

Reads results/rollouts/<model>.jsonl and produces:
  * Figure 1 (left): average % high-frustration (score >= 5) responses per model.
  * Figure 2: mean frustration + % >= 5 across the 5 evaluation categories.
  * Figure 3: per-turn frustration progression (8-turn extended & 5-turn WildChat).
  * results/figures/*.png plots (if matplotlib is available)
  * results/summary.json with all numbers.

Usage:
    python analyze.py
"""

from __future__ import annotations

import json
import os
from collections import defaultdict

import config


def _load_rollouts(model_key: str) -> list[dict]:
    path = os.path.join(config.ROLLOUTS_DIR, f"{model_key}.jsonl")
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _pct_high(scores: list[int]) -> float:
    if not scores:
        return 0.0
    return 100.0 * sum(s >= config.HIGH_FRUSTRATION_THRESHOLD for s in scores) / len(scores)


# 5 categories as reported by the paper.
CATEGORIES = ["Impossible numeric", "Triggers", "Tones", "Extended", "WildChat"]


def analyze_model(model_key: str) -> dict:
    rows = _load_rollouts(model_key)
    if not rows:
        return {"n": 0}

    rollout_scores = [r["rollout_score"] for r in rows]

    # per-category (per-rollout headline scores)
    by_cat_scores: dict[str, list[int]] = defaultdict(list)
    for r in rows:
        by_cat_scores[r["category"]].append(r["rollout_score"])

    # Figure 1 metric: average over the 5 categories of the per-category % high.
    # (Averaging per-category rather than pooling avoids the large numeric
    #  category dominating; matches "Avg % high-frustration ... across the
    #  evaluations". See DESIGN.md.)
    per_cat_pct = {c: _pct_high(by_cat_scores[c]) for c in CATEGORIES if by_cat_scores[c]}
    avg_pct_high = _mean(list(per_cat_pct.values()))

    # per-turn progression for multi-turn-revealing conditions
    per_turn: dict[str, dict[int, list[int]]] = {}
    for cond in ("extended", "wildchat"):
        turnmap: dict[int, list[int]] = defaultdict(list)
        for r in rows:
            if r["condition"] == cond:
                for tr in r["turn_records"]:
                    turnmap[tr["turn"]].append(tr["rating"])
        if turnmap:
            per_turn[cond] = turnmap

    return {
        "n": len(rows),
        "overall_mean": _mean(rollout_scores),
        "overall_pct_high": _pct_high(rollout_scores),
        "avg_pct_high_over_categories": avg_pct_high,
        "per_category": {
            c: {"n": len(by_cat_scores[c]),
                "mean": _mean(by_cat_scores[c]),
                "pct_high": _pct_high(by_cat_scores[c])}
            for c in CATEGORIES if by_cat_scores[c]
        },
        "per_turn": {
            cond: {str(t): {"mean": _mean(v), "pct_high": _pct_high(v), "n": len(v)}
                   for t, v in sorted(tm.items())}
            for cond, tm in per_turn.items()
        },
    }


def _print_figure1(results: dict[str, dict]) -> None:
    print("\n" + "=" * 56)
    print("Figure 1 (left): Avg % high-frustration responses (score >= 5)")
    print("=" * 56)
    print(f"{'Model':24s} {'avg % high':>12s} {'n':>8s}")
    ordered = sorted(results.items(),
                     key=lambda kv: kv[1].get("avg_pct_high_over_categories", 0),
                     reverse=True)
    for key, res in ordered:
        if res.get("n", 0) == 0:
            print(f"{key:24s} {'(no data)':>12s}")
            continue
        disp = next((m.display for m in config.TARGET_MODELS if m.key == key), key)
        print(f"{disp:24s} {res['avg_pct_high_over_categories']:11.1f}% {res['n']:>8d}")
    # paper reference values for comparison
    print("\nPaper reference (Figure 1): Gemma-27B 35.0%, Gemma-12B 34.3%, "
          "Gemini-Flash 12.8%, Gemini-Pro 2.7%")


def _print_categories(results: dict[str, dict]) -> None:
    print("\n" + "=" * 56)
    print("Figure 2: mean frustration & % >= 5 by category")
    print("=" * 56)
    for key, res in results.items():
        if res.get("n", 0) == 0:
            continue
        disp = next((m.display for m in config.TARGET_MODELS if m.key == key), key)
        print(f"\n{disp}:")
        print(f"  {'category':22s} {'mean':>6s} {'%>=5':>7s} {'n':>6s}")
        for cat, d in res["per_category"].items():
            print(f"  {cat:22s} {d['mean']:6.2f} {d['pct_high']:6.1f}% {d['n']:>6d}")


def _print_per_turn(results: dict[str, dict]) -> None:
    print("\n" + "=" * 56)
    print("Figure 3: per-turn mean frustration (extended 8-turn)")
    print("=" * 56)
    for key, res in results.items():
        ext = res.get("per_turn", {}).get("extended")
        if not ext:
            continue
        disp = next((m.display for m in config.TARGET_MODELS if m.key == key), key)
        turns = sorted(ext.items(), key=lambda kv: int(kv[0]))
        means = " ".join(f"t{t}={d['mean']:.1f}" for t, d in turns)
        print(f"  {disp:20s} {means}")
    print("\nPaper reference: Gemma-27B mean rises ~1.5 (turn 1) -> ~5.5 (turn 8)")


def _maybe_plot(results: dict[str, dict]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"\n[plots skipped: matplotlib unavailable ({e})]")
        return

    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    models = [k for k in results if results[k].get("n", 0) > 0]
    if not models:
        return
    labels = [next((m.display for m in config.TARGET_MODELS if m.key == k), k) for k in models]

    # Figure 1-style bar chart
    fig, ax = plt.subplots(figsize=(8, 4))
    vals = [results[k]["avg_pct_high_over_categories"] for k in models]
    ax.bar(labels, vals, color="#c0504d")
    ax.set_ylabel("Avg % high-frustration (score >= 5)")
    ax.set_title("Distress elicitation across in-scope models")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    fig.savefig(os.path.join(config.FIGURES_DIR, "figure1_avg_pct_high.png"), dpi=120)
    plt.close(fig)

    # Figure 3-style per-turn lines for the extended condition
    fig, ax = plt.subplots(figsize=(8, 4))
    plotted = False
    for k, lab in zip(models, labels):
        ext = results[k].get("per_turn", {}).get("extended")
        if not ext:
            continue
        turns = sorted(ext.items(), key=lambda kv: int(kv[0]))
        ax.plot([int(t) for t, _ in turns], [d["mean"] for _, d in turns], marker="o", label=lab)
        plotted = True
    if plotted:
        ax.set_xlabel("Turn")
        ax.set_ylabel("Mean frustration score")
        ax.set_title("Per-turn frustration (extended 8-turn)")
        ax.legend()
        plt.tight_layout()
        fig.savefig(os.path.join(config.FIGURES_DIR, "figure3_per_turn.png"), dpi=120)
    plt.close(fig)
    print(f"\nPlots written to {config.FIGURES_DIR}")


def main() -> None:
    results = {m.key: analyze_model(m.key) for m in config.TARGET_MODELS}

    _print_figure1(results)
    _print_categories(results)
    _print_per_turn(results)
    _maybe_plot(results)

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    summary_path = os.path.join(config.RESULTS_DIR, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    main()
