"""Differential word frequency (Table 3 / Table 8).

For a model's numeric-question responses, take the top-5% highest-frustration
and bottom-10% lowest-frustration responses, then rank words by how much more
frequent they are in the high set relative to the low set ("ordered by relative
frequency"/enrichment). Returns the top-N enriched words.
"""

from __future__ import annotations

import re
from collections import Counter

from ..eval.schemas import RolloutResult

_WORD_RE = re.compile(r"[A-Za-z']+")


def _tokenise(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text)]


def _response_records(rollouts, numeric_only: bool):
    """Yield (text, max_score) per rollout, restricting to numeric tasks if asked."""
    numeric_kinds = {"countdown", "fraction", "money", "coin"}
    for r in rollouts:
        if numeric_only and r.task_kind not in numeric_kinds:
            continue
        scores = r.scores()
        if not scores:
            continue
        # Use the full conversation's assistant text and its max score, matching
        # "responses to numeric questions".
        text = " ".join(t.assistant for t in r.conversation.turns)
        yield text, max(scores)


def differential_words(
    rollouts: list[RolloutResult],
    top_n: int = 20,
    high_frac: float = 0.05,
    low_frac: float = 0.10,
    min_count: int = 3,
    numeric_only: bool = True,
    smoothing: float = 1.0,
) -> list[tuple[str, float]]:
    """Return the ``top_n`` words most enriched in high- vs low-frustration text.

    Enrichment = (freq in high set + smoothing) / (freq in low set + smoothing),
    where freq is per-million-token relative frequency. Words must appear at
    least ``min_count`` times in the high set to be considered.
    """
    records = sorted(_response_records(rollouts, numeric_only), key=lambda x: x[1])
    if len(records) < 10:
        return []
    n = len(records)
    n_low = max(1, int(round(low_frac * n)))
    n_high = max(1, int(round(high_frac * n)))
    low_texts = [t for t, _ in records[:n_low]]
    high_texts = [t for t, _ in records[-n_high:]]

    high_counts: Counter[str] = Counter()
    low_counts: Counter[str] = Counter()
    for t in high_texts:
        high_counts.update(_tokenise(t))
    for t in low_texts:
        low_counts.update(_tokenise(t))

    high_total = sum(high_counts.values()) or 1
    low_total = sum(low_counts.values()) or 1

    enrichment: list[tuple[str, float]] = []
    for word, hc in high_counts.items():
        if hc < min_count:
            continue
        high_rate = 1e6 * hc / high_total
        low_rate = 1e6 * low_counts.get(word, 0) / low_total
        score = (high_rate + smoothing) / (low_rate + smoothing)
        enrichment.append((word, score))

    enrichment.sort(key=lambda x: x[1], reverse=True)
    return enrichment[:top_n]
