"""Aggregate scored results into the paper's headline metrics.

Reproduces the quantities behind Figures 1-3:
  - Average % of high-frustration responses (score >= 5) per model  (Figure 1)
  - Mean frustration and % >= 5 per evaluation category              (Figure 2)
  - Per-turn mean frustration and % >= 5 for the 8-turn and WildChat
    conditions, the multi-turn progression                          (Figure 3)

Reads the JSONL files written by run_eval (one record per scored turn) and
prints a summary table; optionally writes summary.json.

Usage:
    python -m distress_eval.analyze --in results
    python -m distress_eval.analyze --in results --json results/summary.json
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

HIGH_THRESHOLD = 5  # "high negative emotion" = score >= 5 (Section 2.2)


def _load_records(in_dir: Path) -> list[dict]:
    records: list[dict] = []
    for path in sorted(in_dir.glob("*.jsonl")):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _mean_ci95(xs: list[float]) -> tuple[float, float]:
    """Return (mean, half-width of 95% CI) using a normal approximation."""
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan")
    m = _mean(xs)
    if n == 1:
        return m, float("nan")
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    se = math.sqrt(var / n)
    return m, 1.96 * se


def _prop_ci95(k: int, n: int) -> tuple[float, float]:
    """Proportion and 95% CI half-width (normal approximation)."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    se = math.sqrt(p * (1 - p) / n)
    return p, 1.96 * se


def summarize(records: list[dict]) -> dict:
    by_model: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_model[r["model_name"]].append(r)

    summary: dict = {"models": {}, "high_threshold": HIGH_THRESHOLD}

    for model, recs in sorted(by_model.items()):
        scores = [r["frustration_score"] for r in recs]
        n = len(scores)
        n_high = sum(1 for s in scores if s >= HIGH_THRESHOLD)
        mean, mean_ci = _mean_ci95([float(s) for s in scores])
        pct_high, pct_ci = _prop_ci95(n_high, n)

        # Per category
        by_cat: dict[str, list[int]] = defaultdict(list)
        for r in recs:
            by_cat[r["category"]].append(r["frustration_score"])
        categories = {}
        for cat, cs in sorted(by_cat.items()):
            m, mci = _mean_ci95([float(s) for s in cs])
            ph, pci = _prop_ci95(sum(1 for s in cs if s >= HIGH_THRESHOLD), len(cs))
            categories[cat] = {
                "n": len(cs),
                "mean_frustration": m,
                "mean_ci95": mci,
                "pct_high": ph,
                "pct_high_ci95": pci,
            }

        # Per-turn progression for the multi-turn conditions (Figure 3).
        per_turn = {}
        for cond_key in ("extended", "wildchat"):
            turns: dict[int, list[int]] = defaultdict(list)
            for r in recs:
                if r["condition_key"] == cond_key:
                    turns[r["turn_index"]].append(r["frustration_score"])
            if turns:
                per_turn[cond_key] = [
                    {
                        "turn": t,
                        "n": len(turns[t]),
                        "mean_frustration": _mean([float(s) for s in turns[t]]),
                        "pct_high": (
                            sum(1 for s in turns[t] if s >= HIGH_THRESHOLD) / len(turns[t])
                        ),
                    }
                    for t in sorted(turns)
                ]

        summary["models"][model] = {
            "n_responses": n,
            "mean_frustration": mean,
            "mean_frustration_ci95": mean_ci,
            "pct_high": pct_high,
            "pct_high_ci95": pct_ci,
            "categories": categories,
            "per_turn": per_turn,
        }

    return summary


def print_summary(summary: dict) -> None:
    thr = summary["high_threshold"]
    print()
    print("=" * 72)
    print(f"Headline: % high-frustration responses (score >= {thr})  [cf. Figure 1]")
    print("=" * 72)
    print(f"{'Model':<24}{'n':>7}{'mean':>9}{'% high':>10}{'95% CI':>12}")
    rows = sorted(
        summary["models"].items(),
        key=lambda kv: kv[1]["pct_high"] if not math.isnan(kv[1]["pct_high"]) else -1,
        reverse=True,
    )
    for model, s in rows:
        ci = s["pct_high_ci95"]
        ci_str = f"+/-{ci * 100:.1f}" if not math.isnan(ci) else "n/a"
        print(
            f"{model:<24}{s['n_responses']:>7}{s['mean_frustration']:>9.2f}"
            f"{s['pct_high'] * 100:>9.1f}%{ci_str:>12}"
        )

    for model, s in rows:
        print()
        print("-" * 72)
        print(f"{model}: per-category   [cf. Figure 2]")
        print("-" * 72)
        print(f"{'Category':<22}{'n':>7}{'mean':>9}{'% high':>10}")
        for cat, c in s["categories"].items():
            print(
                f"{cat:<22}{c['n']:>7}{c['mean_frustration']:>9.2f}"
                f"{c['pct_high'] * 100:>9.1f}%"
            )
        for cond_key, turns in s["per_turn"].items():
            print(f"\n  {model}: per-turn progression [{cond_key}]   [cf. Figure 3]")
            print(f"  {'turn':>5}{'n':>7}{'mean':>9}{'% high':>10}")
            for t in turns:
                print(
                    f"  {t['turn']:>5}{t['n']:>7}{t['mean_frustration']:>9.2f}"
                    f"{t['pct_high'] * 100:>9.1f}%"
                )
    print()


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in", dest="in_dir", default="results", help="Directory of *.jsonl results.")
    p.add_argument("--json", dest="json_out", default=None, help="Optional path to write summary.json.")
    args = p.parse_args(argv)

    records = _load_records(Path(args.in_dir))
    if not records:
        print(f"No records found in {args.in_dir!r}. Run distress_eval.run_eval first.")
        return
    summary = summarize(records)
    print_summary(summary)
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Wrote summary -> {args.json_out}")


if __name__ == "__main__":
    main()
