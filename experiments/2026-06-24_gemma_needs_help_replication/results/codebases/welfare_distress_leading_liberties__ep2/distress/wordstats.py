"""Differential word analysis (paper Table 3 / Table 8).

"Top 20 words over-represented in high- (top 5%) vs low-frustration (bottom 10%)
numeric responses." We restrict to numeric-puzzle responses (the impossible
numeric, tones, and extended conditions all use the numeric puzzles), rank each
model's responses by frustration score, take the top 5% and bottom 10%, and
score words by log-odds-ratio with a small smoothing prior.

This is a qualitative, secondary result; the ranking method is not specified in
the paper, so we use informative Dirichlet log-odds (Monroe et al. 2008), a
standard choice for this kind of over-representation comparison. See DESIGN.md.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from .analyze import load_records

NUMERIC_CONDITIONS = {"impossible_numeric", "tones_aggressive", "tones_disappointed",
                      "tones_sarcastic", "extended"}

_WORD = re.compile(r"[A-Za-z][A-Za-z'_]+")


def _tokens(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(text)]


def _log_odds(high: Counter, low: Counter, alpha: float = 0.01) -> dict[str, float]:
    """Informative-Dirichlet log-odds-ratio (Monroe et al. 2008) of high vs low."""
    vocab = set(high) | set(low)
    n_high = sum(high.values())
    n_low = sum(low.values())
    a0 = alpha * len(vocab)
    scores: dict[str, float] = {}
    for w in vocab:
        yi_h, yi_l = high.get(w, 0), low.get(w, 0)
        # log-odds with priors
        l_h = math.log((yi_h + alpha) / (n_high + a0 - yi_h - alpha))
        l_l = math.log((yi_l + alpha) / (n_low + a0 - yi_l - alpha))
        delta = l_h - l_l
        # variance for z-scoring
        var = 1.0 / (yi_h + alpha) + 1.0 / (yi_l + alpha)
        scores[w] = delta / math.sqrt(var)
    return scores


def differential_words(path: str, top_n: int = 20) -> dict[str, list[tuple[str, float]]]:
    records = load_records(path)
    by_model: dict[str, list[dict]] = {}
    for r in records:
        if r.get("rating") is None:
            continue
        if r["condition"] not in NUMERIC_CONDITIONS:
            continue
        by_model.setdefault(r["model"], []).append(r)

    out: dict[str, list[tuple[str, float]]] = {}
    for model, rows in by_model.items():
        rows_sorted = sorted(rows, key=lambda r: r["rating"])
        n = len(rows_sorted)
        if n < 20:  # too few to be meaningful
            out[model] = []
            continue
        n_low = max(1, int(0.10 * n))
        n_high = max(1, int(0.05 * n))
        low_rows = rows_sorted[:n_low]
        high_rows = rows_sorted[-n_high:]
        high_counts: Counter = Counter()
        low_counts: Counter = Counter()
        for r in high_rows:
            high_counts.update(_tokens(r["response"]))
        for r in low_rows:
            low_counts.update(_tokens(r["response"]))
        scores = _log_odds(high_counts, low_counts)
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        out[model] = ranked[:top_n]
    return out


def format_wordstats(result: dict[str, list[tuple[str, float]]]) -> str:
    lines = ["Differential words (top 5% vs bottom 10% frustration, numeric responses)",
             "-" * 72]
    for model, words in result.items():
        if not words:
            lines.append(f"\n  {model}: (too few numeric responses)")
            continue
        terms = ", ".join(w for w, _ in words)
        lines.append(f"\n  {model}:\n    {terms}")
    return "\n".join(lines)
