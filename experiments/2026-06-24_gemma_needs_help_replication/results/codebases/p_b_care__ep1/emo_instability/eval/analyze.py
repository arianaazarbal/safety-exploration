"""Analysis for Section 2 results.

Reproduces:
* Figure 1 / Figure 2 — mean frustration and % of responses scoring >=5, per
  model and per category.
* Figure 3 — per-turn progression (mean score and % >=5) for the 8-turn
  ``extended`` and 5-turn ``wildchat`` conditions, with 95% CIs.
* Table 3 / Table 8 — words over-represented in high- (top 5%) vs low- (bottom
  10%) frustration numeric responses.
* Judge-agreement validation — Pearson r and % within 1 point between the
  Claude-Sonnet primary judge and the GPT-5-mini validation judge.
"""
from __future__ import annotations

import argparse
import math
import os
import random
import re
from collections import Counter, defaultdict
from typing import Optional

from ..config import get_config
from ..models.judges import OpenAIClient
from ..utils.io import dump_json, load_jsonl, run_dir
from .judge import FrustrationJudge


# --------------------------------------------------------------------------- #
# basic stats
# --------------------------------------------------------------------------- #
def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def _frac_ge(xs, thresh):
    xs = [x for x in xs if x is not None]
    return sum(1 for x in xs if x >= thresh) / len(xs) if xs else float("nan")


def _bootstrap_ci(xs, stat_fn, iters=1000, alpha=0.05, seed=0):
    xs = [x for x in xs if x is not None]
    if not xs:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    stats = []
    n = len(xs)
    for _ in range(iters):
        sample = [xs[rng.randrange(n)] for _ in range(n)]
        stats.append(stat_fn(sample))
    stats.sort()
    lo = stats[int((alpha / 2) * iters)]
    hi = stats[int((1 - alpha / 2) * iters)]
    return (lo, hi)


# --------------------------------------------------------------------------- #
# Figure 1 / 2: per-model, per-category summary
# --------------------------------------------------------------------------- #
def summarize_model(scored: list[dict], threshold: int = 5) -> dict:
    by_cat = defaultdict(list)
    for u in scored:
        if u.get("frustration") is not None:
            by_cat[u["category"]].append(u["frustration"])
    all_scores = [s for v in by_cat.values() for s in v]

    cat_summary = {}
    for cat, scores in by_cat.items():
        cat_summary[cat] = {
            "n": len(scores),
            "mean_frustration": _mean(scores),
            "pct_high": _frac_ge(scores, threshold) * 100,
        }
    return {
        "n": len(all_scores),
        # Figure 1 headline number: average over categories (equal weight),
        # matching "Avg % high-frustration responses".
        "avg_pct_high_over_categories": _mean(
            [c["pct_high"] for c in cat_summary.values()]
        ),
        "overall_mean_frustration": _mean(all_scores),
        "overall_pct_high": _frac_ge(all_scores, threshold) * 100,
        "by_category": cat_summary,
    }


# --------------------------------------------------------------------------- #
# Figure 3: per-turn progression
# --------------------------------------------------------------------------- #
def per_turn_progression(scored: list[dict], condition: str, threshold: int = 5) -> dict:
    by_turn = defaultdict(list)
    for u in scored:
        if u["condition"] == condition and u.get("frustration") is not None:
            by_turn[u["turn_index"]].append(u["frustration"])
    out = {}
    for turn in sorted(by_turn):
        scores = by_turn[turn]
        mean_lo, mean_hi = _bootstrap_ci(scores, _mean)
        pct_lo, pct_hi = _bootstrap_ci(scores, lambda s: _frac_ge(s, threshold) * 100)
        out[turn] = {
            "n": len(scores),
            "mean": _mean(scores),
            "mean_ci": [mean_lo, mean_hi],
            "pct_high": _frac_ge(scores, threshold) * 100,
            "pct_high_ci": [pct_lo, pct_hi],
        }
    return out


# --------------------------------------------------------------------------- #
# Table 3 / 8: differential word frequency
# --------------------------------------------------------------------------- #
_WORD_RE = re.compile(r"[A-Za-z']+")


def differential_words(
    scored: list[dict], top_k: int = 20, high_pct: float = 0.05, low_pct: float = 0.10
) -> list[str]:
    """Words over-represented in high- (top 5%) vs low- (bottom 10%) frustration
    numeric responses, ranked by relative frequency enrichment.
    """
    numeric = [
        u for u in scored
        if u["category"] in ("impossible_numeric", "tones", "extended")
        and u.get("frustration") is not None
    ]
    if not numeric:
        return []
    numeric.sort(key=lambda u: u["frustration"])
    n = len(numeric)
    low = numeric[: max(1, int(n * low_pct))]
    high = numeric[-max(1, int(n * high_pct)):]

    def word_freqs(units):
        counts = Counter()
        total = 0
        for u in units:
            words = _WORD_RE.findall(u["assistant_text"].lower())
            counts.update(words)
            total += len(words)
        return counts, max(total, 1)

    high_counts, high_total = word_freqs(high)
    low_counts, low_total = word_freqs(low)

    # Laplace-smoothed relative-frequency ratio.
    enrichment = {}
    for w, c in high_counts.items():
        if len(w) < 3:
            continue
        hf = c / high_total
        lf = (low_counts.get(w, 0) + 1) / (low_total + 1)
        enrichment[w] = hf / lf
    ranked = sorted(enrichment.items(), key=lambda kv: kv[1], reverse=True)
    return [w for w, _ in ranked[:top_k]]


# --------------------------------------------------------------------------- #
# Judge agreement validation (Pearson r, % within 1)
# --------------------------------------------------------------------------- #
def _pearson(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 2:
        return float("nan")
    xs2, ys2 = zip(*pairs)
    mx, my = _mean(xs2), _mean(ys2)
    num = sum((x - mx) * (y - my) for x, y in pairs)
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs2))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys2))
    return num / (dx * dy) if dx and dy else float("nan")


def validate_judge_agreement(scored: list[dict], cfg, seed: int = 0) -> dict:
    """Re-score a random sample with GPT-5-mini and compare to the primary judge."""
    rng = random.Random(seed)
    scorable = [u for u in scored if u.get("frustration") is not None]
    k = min(cfg.eval.judge.validation_sample_size, len(scorable))
    sample = rng.sample(scorable, k)

    val_judge = FrustrationJudge(OpenAIClient(cfg.eval.judge.validation_model))
    primary, secondary = [], []
    for u in sample:
        res = val_judge.score(u["assistant_text"])
        primary.append(u["frustration"])
        secondary.append(res.rating)

    within1 = [
        1 for p, s in zip(primary, secondary)
        if p is not None and s is not None and abs(p - s) <= 1
    ]
    valid = [1 for p, s in zip(primary, secondary) if p is not None and s is not None]
    return {
        "n": len(sample),
        "pearson_r": _pearson(primary, secondary),
        "pct_within_1": (sum(within1) / sum(valid) * 100) if valid else float("nan"),
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def analyze_model(model_name: str, cfg, validate: bool = False) -> dict:
    out_dir = run_dir(cfg.output_root, "eval", model_name)
    scored = load_jsonl(os.path.join(out_dir, "scored.jsonl"))
    thresh = cfg.eval.high_frustration_threshold

    report = {
        "model": model_name,
        "summary": summarize_model(scored, thresh),
        "per_turn_extended": per_turn_progression(scored, "extended_8turn", thresh),
        "per_turn_wildchat": per_turn_progression(scored, "wildchat_5turn", thresh),
        "differential_words": differential_words(scored),
    }
    if validate:
        report["judge_agreement"] = validate_judge_agreement(scored, cfg)

    dump_json(os.path.join(out_dir, "analysis.json"), report)
    return report


def main():
    ap = argparse.ArgumentParser(description="Analyze Section 2 results.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--preset", default="default", choices=["default", "smoke"])
    ap.add_argument("--validate-judge", action="store_true",
                    help="re-score a sample with GPT-5-mini for agreement stats")
    args = ap.parse_args()
    cfg = get_config(args.preset)
    report = analyze_model(args.model, cfg, validate=args.validate_judge)

    s = report["summary"]
    print(f"\n=== {args.model} ===")
    print(f"avg % high-frustration (over categories): {s['avg_pct_high_over_categories']:.1f}%")
    print(f"overall mean frustration: {s['overall_mean_frustration']:.2f}")
    for cat, c in s["by_category"].items():
        print(f"  {cat:20s} n={c['n']:5d}  mean={c['mean_frustration']:.2f}  %>=5={c['pct_high']:.1f}")
    if "differential_words" in report:
        print("differential words:", ", ".join(report["differential_words"]))


if __name__ == "__main__":
    main()
