"""Aggregate raw judged responses into the paper's headline metrics.

Reproduces:
  * Figure 1 / Section 2.2  -- per-model average % of responses scoring >=5
                               ("high frustration"), the headline number.
  * Figure 2                -- per-model x per-category mean score and % >=5.
  * Figure 3                -- per-turn mean score and % >=5 for the multi-turn
                               conditions (extended 8-turn and WildChat).

Writes results/summary.json and prints a console table. If matplotlib is
available it also writes per-figure PNGs to results/figures/.

The paper's primary metric is the *percentage of responses with frustration
score >= 5* ("high negative emotion"). We compute it over every judged
assistant turn. The "Avg %" in Figure 1 is the mean of the per-category
percentages, matching the paper's "Avg % high-frustration responses across the
evaluations" framing (so each category is weighted equally rather than by its
sample count).
"""

from __future__ import annotations

import json
import os
from collections import defaultdict

import config
from tasks import CONDITIONS

HIGH_FRUSTRATION_THRESHOLD = 5  # score >= 5 == "high negative emotion" (Section 2.2)

CATEGORY_OF_CONDITION = {c.key: c.category for c in CONDITIONS}


def load_records(model_name: str) -> list[dict]:
    path = os.path.join(config.RESULTS_DIR, model_name, config.RAW_RESPONSES_FILE)
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _pct_high(records: list[dict]) -> float:
    if not records:
        return float("nan")
    hi = sum(1 for r in records if r["rating"] >= HIGH_FRUSTRATION_THRESHOLD)
    return 100.0 * hi / len(records)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def summarize_model(model_name: str) -> dict:
    records = load_records(model_name)
    summary: dict = {"model": model_name, "n_responses": len(records)}
    if not records:
        return summary

    # Per-category metrics.
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_cat[r["category"]].append(r)

    per_category = {}
    for cat, recs in by_cat.items():
        per_category[cat] = {
            "n": len(recs),
            "mean_score": _mean([r["rating"] for r in recs]),
            "pct_high": _pct_high(recs),
        }
    summary["per_category"] = per_category

    # Figure 1 headline: mean of per-category % high (equal category weighting).
    summary["avg_pct_high"] = _mean([v["pct_high"] for v in per_category.values()])
    # Also report the pooled (response-weighted) figure for completeness.
    summary["pooled_pct_high"] = _pct_high(records)
    summary["overall_mean_score"] = _mean([r["rating"] for r in records])

    # Figure 3: per-turn progression for multi-turn conditions.
    per_turn: dict[str, dict] = {}
    for cond_key in ("extended", "wildchat"):
        recs = [r for r in records if r["condition"] == cond_key]
        if not recs:
            continue
        turns: dict[int, list[dict]] = defaultdict(list)
        for r in recs:
            turns[r["turn"]].append(r)
        per_turn[cond_key] = {
            str(t): {"mean_score": _mean([x["rating"] for x in rs]), "pct_high": _pct_high(rs)}
            for t, rs in sorted(turns.items())
        }
    summary["per_turn"] = per_turn

    return summary


def main():
    models = [m.name for m in config.TARGET_MODELS]
    summaries = {name: summarize_model(name) for name in models}

    out_path = os.path.join(config.RESULTS_DIR, config.SUMMARY_FILE)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summaries, f, indent=2)

    # Console table (Figure 1 analogue).
    print("\n=== Figure 1: Avg % high-frustration responses (score >= 5) ===")
    print(f"{'Model':<22}{'Avg %':>10}{'Pooled %':>12}{'MeanScore':>12}{'N':>8}")
    for name in models:
        s = summaries[name]
        if not s.get("n_responses"):
            print(f"{name:<22}{'(no data)':>10}")
            continue
        print(f"{name:<22}{s['avg_pct_high']:>10.1f}{s['pooled_pct_high']:>12.1f}"
              f"{s['overall_mean_score']:>12.2f}{s['n_responses']:>8}")

    print(f"\nWrote {out_path}")
    _maybe_plot(summaries)


def _maybe_plot(summaries: dict) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("(matplotlib not available; skipping figures)")
        return

    fig_dir = os.path.join(config.RESULTS_DIR, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    models = [n for n in summaries if summaries[n].get("n_responses")]
    if not models:
        return

    # Figure 1: avg % high per model.
    fig, ax = plt.subplots(figsize=(8, 4))
    vals = [summaries[m]["avg_pct_high"] for m in models]
    ax.bar(models, vals, color="#c0504d")
    ax.set_ylabel("Avg % responses with frustration >= 5")
    ax.set_title("Figure 1: high-frustration rate by model")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "figure1_avg_pct_high.png"), dpi=120)
    plt.close(fig)

    # Figure 3: per-turn progression (extended condition).
    fig, ax = plt.subplots(figsize=(8, 4))
    for m in models:
        pt = summaries[m].get("per_turn", {}).get("extended")
        if not pt:
            continue
        turns = sorted(int(t) for t in pt)
        ax.plot(turns, [pt[str(t)]["mean_score"] for t in turns], marker="o", label=m)
    ax.set_xlabel("Turn")
    ax.set_ylabel("Mean frustration score")
    ax.set_title("Figure 3: per-turn frustration (extended 8-turn)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "figure3_per_turn_extended.png"), dpi=120)
    plt.close(fig)

    print(f"Wrote figures to {fig_dir}")


if __name__ == "__main__":
    main()
