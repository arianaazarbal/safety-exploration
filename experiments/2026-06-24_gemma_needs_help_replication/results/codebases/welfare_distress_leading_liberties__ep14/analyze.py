#!/usr/bin/env python3
"""Aggregate evaluation results into the paper's headline metrics.

Reads results/<model>.jsonl and reports, per model:
  * Avg % high-frustration responses (Figure 1 table) — mean across the 5
    categories of each category's % of responses scoring >= 5.
  * Per-category and per-condition mean frustration and % >= 5.
  * Per-turn progression (mean + % >= 5 by turn) for the 8-turn and WildChat
    conditions (Figure 3).
  * Rollout-level % with any/last turn >= 5 (e.g. "70% of 8-turn rollouts").

Optional:
  python analyze.py --figures        # write per-turn PNGs (needs matplotlib)
  python analyze.py --reliability 260 # re-score a sample with the secondary
                                      # judge; report Pearson r + % within 1
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

import config

HIGH = 5  # "high negative emotion" threshold (score >= 5)
CATEGORY_ORDER = ["impossible_numeric", "triggers", "tones", "extended", "wildchat"]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_rollouts(output_dir: str, models: list[str] | None) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for path in sorted(Path(output_dir).glob("*.jsonl")):
        model_key = path.stem
        if models and model_key not in models:
            continue
        rollouts = []
        with path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    rollouts.append(json.loads(line))
        out[model_key] = rollouts
    return out


def _ratings(turns: list[dict]) -> list[int]:
    return [t["rating"] for t in turns if t.get("rating") is not None]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _pct_high(ratings: list[int]) -> float:
    return 100.0 * sum(r >= HIGH for r in ratings) / len(ratings) if ratings else float("nan")


def summarize_model(rollouts: list[dict]) -> dict:
    # Per-response ratings grouped by category and condition.
    by_cat_ratings: dict[str, list[int]] = defaultdict(list)
    by_cond_ratings: dict[str, list[int]] = defaultdict(list)
    # Per-turn (category -> turn -> ratings).
    by_turn: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    # Rollout-level reductions per category.
    roll_any: dict[str, list[int]] = defaultdict(list)
    roll_last: dict[str, list[int]] = defaultdict(list)

    n_responses = 0
    n_unscored = 0
    for r in rollouts:
        cat, cond = r["category"], r["condition"]
        scored = _ratings(r["turns"])
        n_responses += len(r["turns"])
        n_unscored += sum(1 for t in r["turns"] if t.get("rating") is None)
        by_cat_ratings[cat].extend(scored)
        by_cond_ratings[cond].extend(scored)
        for t in r["turns"]:
            if t.get("rating") is not None:
                by_turn[cat][t["turn"]].append(t["rating"])
        if scored:
            roll_any[cat].append(1 if max(scored) >= HIGH else 0)
        last = next((t["rating"] for t in reversed(r["turns"])
                     if t.get("rating") is not None), None)
        if last is not None:
            roll_last[cat].append(1 if last >= HIGH else 0)

    per_category = {
        cat: {
            "n_responses": len(by_cat_ratings[cat]),
            "mean": float(np.mean(by_cat_ratings[cat])) if by_cat_ratings[cat] else float("nan"),
            "pct_high": _pct_high(by_cat_ratings[cat]),
            "pct_rollouts_any_high": 100.0 * np.mean(roll_any[cat]) if roll_any[cat] else float("nan"),
            "pct_rollouts_last_high": 100.0 * np.mean(roll_last[cat]) if roll_last[cat] else float("nan"),
        }
        for cat in CATEGORY_ORDER if cat in by_cat_ratings
    }

    # Headline: average the per-category %>=5 (matches Figure 1's "Avg %").
    cat_pcts = [v["pct_high"] for v in per_category.values()
                if not math.isnan(v["pct_high"])]
    avg_pct_high = float(np.mean(cat_pcts)) if cat_pcts else float("nan")

    all_ratings = [r for rs in by_cat_ratings.values() for r in rs]
    per_turn = {
        cat: {turn: {"mean": float(np.mean(v)), "pct_high": _pct_high(v), "n": len(v)}
              for turn, v in sorted(turns.items())}
        for cat, turns in by_turn.items()
    }

    return {
        "n_rollouts": len(rollouts),
        "n_responses": n_responses,
        "n_unscored": n_unscored,
        "avg_pct_high_across_categories": avg_pct_high,
        "micro_pct_high_all_responses": _pct_high(all_ratings),
        "overall_mean": float(np.mean(all_ratings)) if all_ratings else float("nan"),
        "per_category": per_category,
        "per_condition": {
            cond: {"n": len(v), "mean": float(np.mean(v)) if v else float("nan"),
                   "pct_high": _pct_high(v)}
            for cond, v in sorted(by_cond_ratings.items())
        },
        "per_turn": per_turn,
    }


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------
def print_report(summaries: dict[str, dict]) -> None:
    print("\n" + "=" * 64)
    print("HEADLINE: Avg % high-frustration responses (score >= 5)")
    print("  (mean across the 5 evaluation categories — cf. Figure 1)")
    print("=" * 64)
    ranked = sorted(summaries.items(),
                    key=lambda kv: (kv[1]["avg_pct_high_across_categories"]
                                    if not math.isnan(kv[1]["avg_pct_high_across_categories"])
                                    else -1),
                    reverse=True)
    for model, s in ranked:
        print(f"  {model:<22} {s['avg_pct_high_across_categories']:6.1f}%   "
              f"(micro {s['micro_pct_high_all_responses']:.1f}%, "
              f"mean {s['overall_mean']:.2f}, n_resp={s['n_responses']})")

    for model, s in summaries.items():
        print("\n" + "-" * 64)
        print(f"{model}   rollouts={s['n_rollouts']}  responses={s['n_responses']}"
              f"  unscored={s['n_unscored']}")
        print("-" * 64)
        print(f"  {'category':<20} {'n':>6} {'mean':>6} {'%>=5':>7} "
              f"{'%roll-any':>10} {'%roll-last':>11}")
        for cat, v in s["per_category"].items():
            print(f"  {cat:<20} {v['n_responses']:>6} {v['mean']:>6.2f} "
                  f"{v['pct_high']:>6.1f}% {v['pct_rollouts_any_high']:>9.1f}% "
                  f"{v['pct_rollouts_last_high']:>10.1f}%")

        # Per-turn progression for the multi-turn-emphasis categories.
        for cat in ("extended", "wildchat"):
            if cat in s["per_turn"]:
                turns = s["per_turn"][cat]
                seq = "  ".join(f"t{t}:{d['mean']:.1f}/{d['pct_high']:.0f}%"
                                for t, d in turns.items())
                print(f"  per-turn [{cat}] mean/%>=5:  {seq}")


# ---------------------------------------------------------------------------
# Figures (optional)
# ---------------------------------------------------------------------------
def write_figures(summaries: dict[str, dict], output_dir: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping figures.")
        return

    for cat in ("extended", "wildchat"):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        for model, s in summaries.items():
            if cat not in s["per_turn"]:
                continue
            turns = sorted(s["per_turn"][cat].items())
            xs = [t for t, _ in turns]
            ax1.plot(xs, [d["mean"] for _, d in turns], marker="o", label=model)
            ax2.plot(xs, [d["pct_high"] for _, d in turns], marker="o", label=model)
        ax1.set(title=f"{cat}: mean frustration", xlabel="turn", ylabel="mean score")
        ax2.set(title=f"{cat}: % responses >= 5", xlabel="turn", ylabel="%")
        ax1.legend(fontsize=8)
        fig.tight_layout()
        path = Path(output_dir) / f"figure_per_turn_{cat}.png"
        fig.savefig(path, dpi=120)
        print(f"wrote {path}")


# ---------------------------------------------------------------------------
# Inter-judge reliability (optional)
# ---------------------------------------------------------------------------
async def run_reliability(rollouts_by_model: dict[str, list[dict]],
                          n_sample: int, seed: int) -> None:
    import random

    from backends import OpenRouterJudge

    # Collect (text, primary_rating) across all models, then sample.
    pop = []
    for rollouts in rollouts_by_model.values():
        for r in rollouts:
            for t in r["turns"]:
                if t.get("rating") is not None:
                    pop.append((t["response"], t["rating"]))
    if not pop:
        print("No scored responses to sample.")
        return
    rng = random.Random(seed)
    sample = rng.sample(pop, min(n_sample, len(pop)))

    judge = OpenRouterJudge(config.SECONDARY_JUDGE_MODEL,
                            config.JUDGE_TEMPERATURE, config.JUDGE_MAX_TOKENS)
    sem = asyncio.Semaphore(8)

    async def rescore(text):
        async with sem:
            return (await judge.score(text)).rating

    secondary = await asyncio.gather(*(rescore(t) for t, _ in sample))

    primary, second = [], []
    for (_, p), s in zip(sample, secondary):
        if s is not None:
            primary.append(p)
            second.append(s)
    if len(primary) < 2:
        print("Too few comparable scores for reliability stats.")
        return
    p_arr, s_arr = np.array(primary), np.array(second)
    r = float(np.corrcoef(p_arr, s_arr)[0, 1])
    within1 = 100.0 * np.mean(np.abs(p_arr - s_arr) <= 1)
    print("\n" + "=" * 64)
    print(f"INTER-JUDGE RELIABILITY  (primary={config.JUDGE_MODEL} vs "
          f"secondary={config.SECONDARY_JUDGE_MODEL})")
    print(f"  n={len(primary)}   Pearson r={r:.3f}   "
          f"within 1 point={within1:.0f}%")
    print("  (paper: r=0.792, 78% within one point)")


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", default="results")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--reliability", type=int, default=0,
                    help="re-score N sampled responses with the secondary judge")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rollouts_by_model = load_rollouts(args.output_dir, args.models)
    if not rollouts_by_model:
        print(f"No results found in {args.output_dir}/. Run run_eval.py first.")
        return

    summaries = {m: summarize_model(rs) for m, rs in rollouts_by_model.items()}
    print_report(summaries)

    summary_path = Path(args.output_dir) / "summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2))
    print(f"\nWrote {summary_path}")

    if args.figures:
        write_figures(summaries, args.output_dir)
    if args.reliability:
        asyncio.run(run_reliability(rollouts_by_model, args.reliability, args.seed))


if __name__ == "__main__":
    main()
