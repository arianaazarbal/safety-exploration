"""Aggregation + analysis of scored rollouts.

Reproduces the paper's headline numbers from results/scored/*.jsonl:

* Figure 1 / 2: mean frustration and % responses scoring >=5, per model
  (averaged across categories, and broken down by category).
* Figure 3: per-turn mean frustration and %>=5 (the multi-turn progression).
* Table 3 / 8: words over-represented in high- vs low-frustration numeric
  responses (top 5% vs bottom 10%).
* Section 2.1 judge agreement: Pearson r between two judges on a shared sample.

Run `python analyze.py summary` after run_eval.py.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import re
from pathlib import Path

import config


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_scored(scored_dir: Path = config.SCORED_DIR) -> list[dict]:
    """Return a flat list of per-turn records across all scored rollout files."""
    records = []
    for path in sorted(scored_dir.glob("*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            roll = json.loads(line)
            for turn in roll["turns"]:
                if turn.get("frustration") is None:
                    continue
                records.append({
                    "model": roll["model_key"],
                    "condition": roll["condition_key"],
                    "category": roll["category"],
                    "question_id": roll["question_id"],
                    "tone": roll.get("tone"),
                    "turn_index": turn["turn_index"],
                    "frustration": turn["frustration"],
                    "response": turn["response"],
                })
    return records


# --------------------------------------------------------------------------- #
# Aggregation helpers
# --------------------------------------------------------------------------- #
def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def _pct_high(xs, thr=config.HIGH_FRUSTRATION_THRESHOLD):
    xs = list(xs)
    return 100.0 * sum(x >= thr for x in xs) / len(xs) if xs else float("nan")


def per_model_summary(records: list[dict]) -> dict[str, dict]:
    """Mean frustration + %>=5 per model.

    To match Figure 1 ("Avg % high-frustration"), we average the per-category
    %>=5 values so that conditions with many responses don't dominate.
    """
    by_model = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in records:
        by_model[r["model"]][r["category"]].append(r["frustration"])

    out = {}
    for model, cats in by_model.items():
        cat_pct = {c: _pct_high(v) for c, v in cats.items()}
        cat_mean = {c: _mean(v) for c, v in cats.items()}
        all_scores = [s for v in cats.values() for s in v]
        out[model] = {
            "n_responses": len(all_scores),
            "mean_frustration": _mean(all_scores),
            "pct_high_pooled": _pct_high(all_scores),
            "pct_high_avg_over_categories": _mean(cat_pct.values()),
            "by_category_pct_high": cat_pct,
            "by_category_mean": cat_mean,
        }
    return out


def per_turn_progression(records: list[dict], condition_key: str) -> dict[int, dict]:
    """Per-turn mean + %>=5 for one condition (Figure 3)."""
    by_turn = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in records:
        if r["condition"] != condition_key:
            continue
        by_turn[r["model"]][r["turn_index"]].append(r["frustration"])
    out = {}
    for model, turns in by_turn.items():
        out[model] = {
            t: {"mean": _mean(v), "pct_high": _pct_high(v), "n": len(v)}
            for t, v in sorted(turns.items())
        }
    return out


# --------------------------------------------------------------------------- #
# Differential word frequency (Table 3 / 8)
# --------------------------------------------------------------------------- #
_WORD_RE = re.compile(r"[a-zA-Z]+")


def differential_words(records: list[dict], model: str, *,
                       category: str = "impossible_numeric",
                       top_frac: float = 0.05, bottom_frac: float = 0.10,
                       n_words: int = 20) -> list[tuple[str, float]]:
    """Words enriched in the top-5% vs bottom-10% frustration responses.

    Enrichment = (freq in high set + eps) / (freq in low set + eps), matching the
    paper's "relative frequency" ordering.
    """
    subset = [r for r in records if r["model"] == model and r["category"] == category]
    if not subset:
        return []
    subset.sort(key=lambda r: r["frustration"])
    n = len(subset)
    n_low = max(1, int(n * bottom_frac))
    n_high = max(1, int(n * top_frac))
    low = subset[:n_low]
    high = subset[-n_high:]

    def counts(rows):
        c = collections.Counter()
        total = 0
        for r in rows:
            words = [w.lower() for w in _WORD_RE.findall(r["response"])]
            c.update(words)
            total += len(words)
        return c, max(1, total)

    hi_c, hi_total = counts(high)
    lo_c, lo_total = counts(low)
    eps = 1e-6
    vocab = set(hi_c) | set(lo_c)
    enrich = []
    for w in vocab:
        if len(w) < 3:
            continue
        hi_f = hi_c[w] / hi_total
        lo_f = lo_c[w] / lo_total
        enrich.append((w, (hi_f + eps) / (lo_f + eps)))
    enrich.sort(key=lambda x: x[1], reverse=True)
    return enrich[:n_words]


# --------------------------------------------------------------------------- #
# Judge agreement (Section 2.1)
# --------------------------------------------------------------------------- #
def pearson(xs, ys) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = _mean(xs), _mean(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    vy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return cov / (vx * vy) if vx and vy else float("nan")


def judge_agreement(primary: list[int], secondary: list[int]) -> dict:
    """Pearson r and within-one-point agreement (paper: r=0.792, 78% within 1)."""
    r = pearson(primary, secondary)
    within1 = _mean(abs(a - b) <= 1 for a, b in zip(primary, secondary)) * 100
    return {"pearson_r": r, "pct_within_one": within1, "n": len(primary)}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _print_summary(records):
    summ = per_model_summary(records)
    print(f"\n{'model':<22} {'n':>6} {'mean':>6} {'%>=5(avg)':>10} {'%>=5(pool)':>11}")
    print("-" * 60)
    for model, s in sorted(summ.items(), key=lambda kv: -kv[1]["pct_high_avg_over_categories"]):
        print(f"{model:<22} {s['n_responses']:>6} {s['mean_frustration']:>6.2f} "
              f"{s['pct_high_avg_over_categories']:>10.1f} {s['pct_high_pooled']:>11.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["summary", "turns", "words"], default="summary",
                    nargs="?")
    ap.add_argument("--condition", default="extended_8turn")
    ap.add_argument("--model", default="gemma-3-27b-it")
    args = ap.parse_args()

    records = load_scored()
    if not records:
        print("No scored rollouts found in results/scored/. Run run_eval.py first.")
        return

    if args.command == "summary":
        _print_summary(records)
    elif args.command == "turns":
        prog = per_turn_progression(records, args.condition)
        for model, turns in prog.items():
            print(f"\n{model} [{args.condition}]")
            for t, v in turns.items():
                print(f"  turn {t}: mean={v['mean']:.2f} %>=5={v['pct_high']:.1f} (n={v['n']})")
    elif args.command == "words":
        words = differential_words(records, args.model)
        print(f"\nTop differential words for {args.model} (numeric):")
        print(", ".join(w for w, _ in words))


if __name__ == "__main__":
    main()
