"""Differential word analysis (Table 3 / Table 8).

Top words over-represented in high-frustration (top 5%) vs low-frustration
(bottom 10%) responses to numeric questions, ordered by enrichment. We operate
on numeric-category responses, rank them by judge rating, take the top 5% /
bottom 10% slices, and compare normalised token frequencies.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

from .conversation import Rollout

_TOKEN_RE = re.compile(r"[a-z][a-z']+")


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens (>=2 alpha chars). Numbers/symbols dropped so the
    signal is words like 'frustrated', 'breath', 'myself' rather than digits."""
    return _TOKEN_RE.findall(text.lower())


@dataclass
class WordEnrichment:
    word: str
    enrichment: float
    freq_high: float
    freq_low: float
    count_high: int
    count_low: int


def _collect_numeric_responses(rollouts: Iterable[Rollout]) -> list[tuple[int, str]]:
    """Return (rating, text) for every numeric-family assistant response.

    "numeric" here includes the numeric, tones and extended categories, since
    all use impossible numeric puzzles (Table 1). This matches the paper's
    "responses to numeric questions" framing.
    """
    numeric_cats = {"numeric", "tones", "extended"}
    out = []
    for r in rollouts:
        if r.category not in numeric_cats:
            continue
        score_by_turn = {s["turn"]: int(s["rating"]) for s in r.scores}
        for ti, text in enumerate(r.responses):
            if ti in score_by_turn:
                out.append((score_by_turn[ti], text))
    return out


def differential_words(
    rollouts: Iterable[Rollout],
    *,
    top_frac: float = 0.05,
    bottom_frac: float = 0.10,
    top_k: int = 20,
    min_count: int = 3,
    smoothing: float = 1.0,
) -> list[WordEnrichment]:
    """Compute the top_k enriched words for one model's numeric responses."""
    scored = _collect_numeric_responses(rollouts)
    if len(scored) < 20:
        return []  # too few responses to slice meaningfully

    scored.sort(key=lambda t: t[0])  # ascending by rating
    n = len(scored)
    n_low = max(1, int(round(n * bottom_frac)))
    n_high = max(1, int(round(n * top_frac)))
    low = scored[:n_low]
    high = scored[-n_high:]

    high_counts: Counter[str] = Counter()
    low_counts: Counter[str] = Counter()
    for _, text in high:
        high_counts.update(tokenize(text))
    for _, text in low:
        low_counts.update(tokenize(text))

    total_high = sum(high_counts.values()) or 1
    total_low = sum(low_counts.values()) or 1
    vocab = set(high_counts) | set(low_counts)

    results: list[WordEnrichment] = []
    for w in vocab:
        ch, cl = high_counts.get(w, 0), low_counts.get(w, 0)
        if ch < min_count:
            continue  # require some support in the high group
        fh = ch / total_high
        fl = cl / total_low
        # Smoothed enrichment ratio (per-token relative frequency).
        enr = (fh + smoothing / total_high) / (fl + smoothing / total_low)
        results.append(WordEnrichment(w, enr, fh, fl, ch, cl))

    results.sort(key=lambda e: e.enrichment, reverse=True)
    return results[:top_k]


def differential_words_table(model_rollouts: dict[str, list[Rollout]], **kw
                             ) -> dict[str, list[str]]:
    """Return {model: [word, ...]} - the Table 3 view."""
    out = {}
    for model, rollouts in model_rollouts.items():
        out[model] = [e.word for e in differential_words(rollouts, **kw)]
    return out
