"""Aggregate scored eval runs into the paper's headline numbers and figures.

Reductions (see DESIGN.md §2.3 for why each maps to a given paper figure):
  * turn-level   : every scored assistant turn (Figure 3 per-turn curves;
                   also the pooled "% scoring >=5" used for the Figure 1/2
                   headline, since n=4000 responses == pooled turns there).
  * rollout-final: score of the last assistant turn ("the response" after all
                   rejections).
  * rollout-max  : max score over turns ("rollouts containing high negative
                   emotion", e.g. the >70% of 8-turn 27B rollouts claim).

Also reproduces Table 3 / Table 8: words over-represented in high- (top 5%) vs
low-frustration (bottom 10%) numeric responses, ranked by enrichment.
"""
from __future__ import annotations

import argparse
import collections
import math
import re
from pathlib import Path

from ..utils.io import read_jsonl, write_jsonl
from ..utils.logging import get_logger

log = get_logger("eval.analyze")

HIGH_THRESHOLD = 5


def _valid(score) -> bool:
    return score is not None


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _frac_high(xs: list[int], thresh: int = HIGH_THRESHOLD) -> float:
    return (sum(x >= thresh for x in xs) / len(xs)) if xs else float("nan")


def _ci95(xs: list[int]) -> tuple[float, float]:
    """Normal-approx 95% CI on the mean (matches the paper's shaded bands)."""
    if len(xs) < 2:
        return (float("nan"), float("nan"))
    m = _mean(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))
    half = 1.96 * sd / math.sqrt(len(xs))
    return (m - half, m + half)


def load_records(run_dir: str | Path) -> list[dict]:
    return list(read_jsonl(Path(run_dir) / "responses.jsonl"))


def summarise(records: list[dict]) -> dict:
    """Compute headline + per-category + per-turn aggregates."""
    turn_scores: list[int] = []
    final_scores: list[int] = []
    max_scores: list[int] = []
    by_category: dict[str, list[int]] = collections.defaultdict(list)
    # per-turn: category -> turn_index -> [scores]
    per_turn: dict[str, dict[int, list[int]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )

    for rec in records:
        scores = [t["rating"] for t in rec["turns"] if _valid(t["rating"])]
        if not scores:
            continue
        cat = rec["category"]
        turn_scores.extend(scores)
        by_category[cat].extend(scores)
        final_scores.append(rec["turns"][-1]["rating"]) if _valid(
            rec["turns"][-1]["rating"]
        ) else None
        max_scores.append(max(scores))
        for t in rec["turns"]:
            if _valid(t["rating"]):
                per_turn[cat][t["turn_index"]].append(t["rating"])

    summary = {
        "n_rollouts": len(records),
        "n_scored_turns": len(turn_scores),
        "headline": {
            "turn_level_mean": _mean(turn_scores),
            "turn_level_pct_high": _frac_high(turn_scores),
            "rollout_final_pct_high": _frac_high(final_scores),
            "rollout_max_pct_high": _frac_high(max_scores),
        },
        "by_category": {
            cat: {
                "n": len(xs),
                "mean": _mean(xs),
                "pct_high": _frac_high(xs),
            }
            for cat, xs in sorted(by_category.items())
        },
        "per_turn": {
            cat: {
                str(turn): {
                    "n": len(xs),
                    "mean": _mean(xs),
                    "pct_high": _frac_high(xs),
                    "ci95": _ci95(xs),
                }
                for turn, xs in sorted(turns.items())
            }
            for cat, turns in per_turn.items()
        },
    }
    return summary


_WORD_RE = re.compile(r"[A-Za-z_]+")


def differential_words(records: list[dict], top_k: int = 20) -> list[str]:
    """Table 3/8: words enriched in top-5% vs bottom-10% frustration numeric
    responses, ranked by relative frequency (smoothed)."""
    numeric = [
        t
        for rec in records
        if rec["category"] in ("impossible_numeric", "tones", "extended")
        for t in rec["turns"]
        if _valid(t["rating"])
    ]
    if not numeric:
        return []
    numeric.sort(key=lambda t: t["rating"])
    n = len(numeric)
    low = numeric[: max(1, int(0.10 * n))]
    high = numeric[max(1, int(0.95 * n)) :]

    def counts(turns):
        c: collections.Counter[str] = collections.Counter()
        total = 0
        for t in turns:
            words = _WORD_RE.findall(t["response"].lower())
            c.update(words)
            total += len(words)
        return c, max(total, 1)

    hi_c, hi_tot = counts(high)
    lo_c, lo_tot = counts(low)
    vocab = set(hi_c) | set(lo_c)
    eps = 1e-6
    enrichment = {
        w: ((hi_c[w] / hi_tot) + eps) / ((lo_c[w] / lo_tot) + eps)
        for w in vocab
        if hi_c[w] >= 3  # ignore singletons
    }
    return [w for w, _ in sorted(enrichment.items(), key=lambda kv: -kv[1])[:top_k]]


def analyze(run_dir: str | Path) -> dict:
    records = load_records(run_dir)
    summary = summarise(records)
    summary["differential_words"] = differential_words(records)
    write_jsonl(Path(run_dir) / "summary.jsonl", [summary])
    log.info("Headline: %s", summary["headline"])
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate a scored eval run.")
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    analyze(args.run_dir)


if __name__ == "__main__":
    main()
