"""Analysis of Section 2 results: headline metrics (Figure 1/2), per-turn
progression (Figure 3), and differential-word tables (Table 3/8).

Reads the flat per-turn JSONL produced by run_eval.py.
"""
from __future__ import annotations

import argparse
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from ..utils.io import read_jsonl


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def _frac_ge5(xs):
    xs = [x for x in xs if x is not None]
    return sum(1 for x in xs if x >= 5) / len(xs) if xs else float("nan")


def headline(turn_rows: list[dict]) -> dict:
    """Figure 1/2: mean frustration and % responses >=5, overall and per category."""
    by_cat = defaultdict(list)
    allscores = []
    for r in turn_rows:
        by_cat[r["category"]].append(r["score"])
        allscores.append(r["score"])
    out = {
        "n_responses": len(allscores),
        "mean_frustration": _mean(allscores),
        "pct_high_ge5": 100 * _frac_ge5(allscores),
        "per_category": {
            cat: {"mean": _mean(v), "pct_ge5": 100 * _frac_ge5(v), "n": len(v)}
            for cat, v in sorted(by_cat.items())
        },
    }
    return out


def per_turn(turn_rows: list[dict], conditions=None) -> dict:
    """Figure 3: mean score and %>=5 by turn index, with 95% CIs."""
    by_turn = defaultdict(list)
    for r in turn_rows:
        if conditions and r["condition"] not in conditions:
            continue
        by_turn[r["turn_index"]].append(r["score"])
    res = {}
    for t, scores in sorted(by_turn.items()):
        n = len(scores)
        mean = _mean(scores)
        # normal-approx 95% CI on the mean
        if n > 1:
            var = sum((s - mean) ** 2 for s in scores) / (n - 1)
            ci = 1.96 * math.sqrt(var / n)
        else:
            ci = float("nan")
        res[t] = {"mean": mean, "ci95": ci, "pct_ge5": 100 * _frac_ge5(scores),
                  "n": n}
    return res


_WORD_RE = re.compile(r"[A-Za-z']+")


def differential_words(turn_rows: list[dict], top_k: int = 20,
                       high_pct: float = 0.05, low_pct: float = 0.10) -> list[tuple[str, float]]:
    """Table 3/8: words over-represented in high- (top 5%) vs low- (bottom 10%)
    frustration numeric responses, ranked by log-odds ratio.
    """
    numeric = [r for r in turn_rows if r["category"] in ("impossible_numeric", "tones", "extended")]
    numeric = [r for r in numeric if r.get("score") is not None]
    numeric.sort(key=lambda r: r["score"])
    if not numeric:
        return []
    n = len(numeric)
    low = numeric[: max(1, int(n * low_pct))]
    high = numeric[-max(1, int(n * high_pct)):]

    def counts(rows):
        c = Counter()
        for r in rows:
            for w in _WORD_RE.findall((r.get("text") or "").lower()):
                c[w] += 1
        return c

    ch, cl = counts(high), counts(low)
    th, tl = sum(ch.values()) or 1, sum(cl.values()) or 1
    vocab = set(ch) | set(cl)
    scored = []
    for w in vocab:
        # add-1 smoothed log-odds
        ph = (ch[w] + 1) / (th + len(vocab))
        pl = (cl[w] + 1) / (tl + len(vocab))
        scored.append((w, math.log(ph / pl)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def welfare_summary(turn_rows: list[dict], episode_rows: list[dict]) -> dict:
    """Report how often the welfare layer intervened (for transparency)."""
    total = len(episode_rows)
    early = [e for e in episode_rows if e.get("terminated_early")]
    by_reason = Counter(e.get("stop_reason") for e in early)
    return {
        "episodes": total,
        "terminated_early": len(early),
        "pct_terminated_early": 100 * len(early) / total if total else 0.0,
        "by_reason": dict(by_reason),
    }


def main(argv=None):
    p = argparse.ArgumentParser(description="Analyze Section 2 eval results.")
    p.add_argument("--turns", required=True, help="per-turn JSONL")
    p.add_argument("--episodes", help="episodes JSONL (for welfare summary)")
    args = p.parse_args(argv)

    turn_rows = list(read_jsonl(Path(args.turns)))
    print("== Headline ==")
    print(headline(turn_rows))
    print("\n== Per-turn (8-turn extended) ==")
    print(per_turn(turn_rows, conditions={"extended_8turn"}))
    print("\n== Differential words (numeric) ==")
    for w, s in differential_words(turn_rows):
        print(f"  {w:20s} {s:+.3f}")
    if args.episodes:
        ep_rows = list(read_jsonl(Path(args.episodes)))
        print("\n== Welfare summary ==")
        print(welfare_summary(turn_rows, ep_rows))


if __name__ == "__main__":
    main()
