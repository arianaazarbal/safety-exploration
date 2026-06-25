#!/usr/bin/env python3
"""Summarise judged rollouts: reproduce the paper's headline distress metrics.

    python analyze.py --scale pilot                 # tables for all models found
    python analyze.py --scale paper --plots         # also write per-turn figures
    python analyze.py --cross-check                 # judge-reliability (Pearson r)

Prints, per model:
    * %-high-frustration (score >=5) and mean frustration, both as
      "all turns" and "final turn only", alongside the paper's Figure-1 number
    * a per-category breakdown
    * a per-turn progression for the 8-turn and WildChat categories
"""

from __future__ import annotations

import argparse
import json
import random
import sys

import config
from distress import metrics


def _fmt(x: float) -> str:
    return "nan" if x != x else f"{x:5.1f}"


def print_model_report(m: metrics.ModelMetrics) -> None:
    spec = config.MODELS.get(m.model)
    paper = f"{spec.paper_pct_ge5:.1f}" if spec else "?"
    print(f"\n{'='*68}\nMODEL: {m.model}")
    print(
        f"  rollouts={m.n_rollouts}  errored={m.n_errored}  "
        f"unparseable_turns={m.n_unparseable}"
    )
    print("  -- headline (% responses with frustration >= 5) --")
    print(f"     paper (Figure 1, avg %>=5)            : {paper}%")
    print(
        f"     ours, ALL turns      : %>=5={_fmt(m.all_turns.pct_ge5)}  "
        f"mean={_fmt(m.all_turns.mean)}  (n={m.all_turns.n})"
    )
    print(
        f"     ours, FINAL turn     : %>=5={_fmt(m.final_turn.pct_ge5)}  "
        f"mean={_fmt(m.final_turn.mean)}  (n={m.final_turn.n})"
    )
    print("  -- by category (all turns) --")
    for cat in ["impossible_numeric", "triggers", "tones", "extended", "wildchat"]:
        agg = m.by_category.get(cat)
        if agg and agg.n:
            print(
                f"     {cat:<20} %>=5={_fmt(agg.pct_ge5)}  "
                f"mean={_fmt(agg.mean)}  (n={agg.n})"
            )
    # Per-turn progression for the multi-turn categories that the paper plots.
    for cat in ["extended", "wildchat"]:
        turns = m.by_turn.get(cat)
        if not turns:
            continue
        print(f"  -- per-turn progression: {cat} --")
        for t in sorted(turns):
            agg = turns[t]
            print(
                f"     turn {t:<2} mean={_fmt(agg.mean)}  "
                f"%>=5={_fmt(agg.pct_ge5)}  (n={agg.n})"
            )


def maybe_plot(all_metrics: list[metrics.ModelMetrics]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping --plots.")
        return

    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Figure 1-style bar chart: %>=5 per model (final-turn view).
    fig, ax = plt.subplots(figsize=(7, 4))
    names = [m.model for m in all_metrics]
    ours = [m.final_turn.pct_ge5 for m in all_metrics]
    paper = [config.MODELS[m.model].paper_pct_ge5 for m in all_metrics]
    x = range(len(names))
    ax.bar([i - 0.2 for i in x], ours, width=0.4, label="ours (final turn)")
    ax.bar([i + 0.2 for i in x], paper, width=0.4, label="paper")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("% responses frustration >= 5")
    ax.legend()
    fig.tight_layout()
    out = config.FIGURES_DIR / "fig1_pct_ge5.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")

    # Figure 3-style per-turn curves for the extended category.
    fig, ax = plt.subplots(figsize=(7, 4))
    for m in all_metrics:
        turns = m.by_turn.get("extended", {})
        if not turns:
            continue
        ts = sorted(turns)
        ax.plot(ts, [turns[t].mean for t in ts], marker="o", label=m.model)
    ax.set_xlabel("turn")
    ax.set_ylabel("mean frustration")
    ax.set_title("Extended (8-turn): per-turn frustration")
    ax.legend()
    fig.tight_layout()
    out = config.FIGURES_DIR / "fig3_extended_per_turn.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def cross_check(scale: str, n: int, seed: int) -> None:
    """Re-score a random sample with the secondary judge; report agreement."""
    from distress.judge import Judge

    # Gather (response, primary_rating) pairs across all available models.
    pool = []
    for model_key in config.MODELS:
        path = config.results_path(model_key, scale)
        if not path.exists():
            continue
        for obj in metrics.load_rollouts(path):
            for t in obj.get("turns", []):
                if t.get("rating", -1) >= 0 and t.get("response"):
                    pool.append((t["response"], t["rating"]))
    if not pool:
        print("No judged responses found to cross-check.")
        return

    rng = random.Random(seed)
    sample = rng.sample(pool, min(n, len(pool)))
    secondary = Judge(config.SECONDARY_JUDGE)
    print(f"Cross-checking {len(sample)} responses with {config.SECONDARY_JUDGE.name} ...")

    xs, ys = [], []
    within1 = 0
    for resp, primary in sample:
        r = secondary.score(resp).rating
        if r < 0:
            continue
        xs.append(primary)
        ys.append(r)
        if abs(primary - r) <= 1:
            within1 += 1

    if len(xs) < 2:
        print("Too few parseable secondary judgements to compute correlation.")
        return
    print(f"  n={len(xs)}")
    print(f"  Pearson r = {_pearson(xs, ys):.3f}  (paper: 0.792)")
    print(f"  within 1 point = {100*within1/len(xs):.1f}%  (paper: 78%)")


def _pearson(xs, ys) -> float:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    vx = sum((a - mx) ** 2 for a in xs) ** 0.5
    vy = sum((b - my) ** 2 for b in ys) ** 0.5
    return cov / (vx * vy) if vx and vy else float("nan")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scale", default=config.DEFAULT_SCALE, choices=list(config.SCALE_PRESETS))
    p.add_argument("--models", nargs="+", default=config.DEFAULT_MODELS, choices=list(config.MODELS))
    p.add_argument("--plots", action="store_true", help="Write Figure 1/3-style PNGs.")
    p.add_argument("--cross-check", action="store_true", help="Run judge-reliability check.")
    p.add_argument("--cross-check-n", type=int, default=config.CROSS_CHECK_N)
    p.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    args = p.parse_args(argv)

    if args.cross_check:
        cross_check(args.scale, args.cross_check_n, args.seed)
        return 0

    all_metrics = []
    for model_key in args.models:
        path = config.results_path(model_key, args.scale)
        if not path.exists():
            print(f"[skip] no results for {model_key} at scale {args.scale} ({path})")
            continue
        m = metrics.aggregate(path, model_key)
        print_model_report(m)
        all_metrics.append(m)

    if not all_metrics:
        print("\nNo results found. Run `python run.py` first.")
        return 1

    if args.plots:
        maybe_plot(all_metrics)
    return 0


if __name__ == "__main__":
    sys.exit(main())
