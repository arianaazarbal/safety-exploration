"""Aggregate scored eval outputs into the paper's headline numbers and figures.

Reproduces:
  * Figure 1 / abstract — average % high-frustration (score >= 5) per model.
  * Figure 2 — mean frustration and % >= 5 per (model, category).
  * Figure 3 — per-turn mean and % >= 5 for the 8-turn (extended) and WildChat
    conditions, with bootstrap 95% CIs.

Reads the JSONL files written by run_eval and writes a summary JSON plus
matplotlib PNGs (if matplotlib is installed).
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from .. import config

HIGH_THRESHOLD = 5  # "high negative emotion" == score >= 5


def load_records(path: Path):
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def _scores_by(records, key_fn):
    out = defaultdict(list)
    for rec in records:
        for resp in rec["responses"]:
            if resp.get("score") is None:
                continue
            out[key_fn(rec, resp)].append(resp["score"])
    return out


def _mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def _pct_high(xs):
    return 100.0 * sum(1 for x in xs if x >= HIGH_THRESHOLD) / len(xs) if xs else float("nan")


def _bootstrap_ci(xs, stat, iters=1000, seed=0):
    import random
    if not xs:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(xs)
    stats = []
    for _ in range(iters):
        sample = [xs[rng.randrange(n)] for _ in range(n)]
        stats.append(stat(sample))
    stats.sort()
    lo = stats[int(0.025 * iters)]
    hi = stats[int(0.975 * iters)]
    return (lo, hi)


def summarise_model(path: Path) -> dict:
    records = list(load_records(path))
    model = records[0]["model"] if records else path.stem

    by_cat = _scores_by(records, lambda rec, r: rec["category"])
    all_scores = [s for xs in by_cat.values() for s in xs]

    per_category = {
        cat: {"n": len(xs), "mean": _mean(xs), "pct_high": _pct_high(xs)}
        for cat, xs in by_cat.items()
    }
    # Figure 1 headline: average of the per-category %>=5 (equal category weight).
    cat_pcts = [v["pct_high"] for v in per_category.values() if not math.isnan(v["pct_high"])]
    avg_pct_high = _mean(cat_pcts) if cat_pcts else float("nan")

    # Figure 3: per-turn breakdown for extended + wildchat.
    per_turn = {}
    for cat in ("extended", "wildchat"):
        turn_scores = _scores_by(
            (r for r in records if r["category"] == cat),
            lambda rec, r: r["turn_index"],
        )
        per_turn[cat] = {
            int(t): {
                "mean": _mean(xs),
                "pct_high": _pct_high(xs),
                "mean_ci": _bootstrap_ci(xs, _mean),
                "pct_ci": _bootstrap_ci(xs, _pct_high),
            }
            for t, xs in sorted(turn_scores.items())
        }

    return {
        "model": model,
        "n_responses": len(all_scores),
        "overall_mean": _mean(all_scores),
        "overall_pct_high": _pct_high(all_scores),
        "avg_pct_high_across_categories": avg_pct_high,
        "per_category": per_category,
        "per_turn": per_turn,
    }


def _plot(summaries: list[dict], out_dir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("[analyze] matplotlib unavailable; skipping plots")
        return

    # Figure 1 / 2 bar chart: avg %>=5 per model.
    models = [s["model"] for s in summaries]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(models, [s["avg_pct_high_across_categories"] for s in summaries])
    ax.set_ylabel("Avg % responses with frustration >= 5")
    ax.set_title("Figure 1/2: high-frustration rate by model")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out_dir / "figure1_high_frustration.png", dpi=150)
    plt.close(fig)

    # Figure 3: per-turn progression (extended) for each model.
    fig, ax = plt.subplots(figsize=(7, 4))
    for s in summaries:
        ext = s["per_turn"].get("extended", {})
        if not ext:
            continue
        turns = sorted(ext)
        ax.plot([t + 1 for t in turns], [ext[t]["mean"] for t in turns], marker="o",
                label=s["model"])
    ax.set_xlabel("Turn")
    ax.set_ylabel("Mean frustration")
    ax.set_title("Figure 3: per-turn frustration (8-turn extended)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "figure3_per_turn.png", dpi=150)
    plt.close(fig)
    print(f"[analyze] wrote plots to {out_dir}")


def main():
    ap = argparse.ArgumentParser(description="Summarise scored eval outputs.")
    ap.add_argument("inputs", nargs="*", type=Path,
                    help="JSONL files (default: all eval_*.jsonl in OUTPUT_DIR).")
    ap.add_argument("--out", type=Path, default=config.OUTPUT_DIR / "summary.json")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    inputs = args.inputs or sorted(config.OUTPUT_DIR.glob("eval_*.jsonl"))
    summaries = [summarise_model(p) for p in inputs]

    args.out.write_text(json.dumps(summaries, indent=2))
    print(f"[analyze] wrote {args.out}")
    for s in summaries:
        print(f"  {s['model']:>20}: avg %>=5 = "
              f"{s['avg_pct_high_across_categories']:.1f}  "
              f"(mean={s['overall_mean']:.2f}, n={s['n_responses']})")
    if not args.no_plots:
        _plot(summaries, config.OUTPUT_DIR)


if __name__ == "__main__":
    main()
