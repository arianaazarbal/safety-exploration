"""Differential word frequency: words over-represented in high- vs low-frustration responses.

Reproduces Table 3 / Table 8: for a model's numeric-puzzle responses, take the top-5% most
frustrated and bottom-10% least frustrated responses (by judge score), and rank words by how
much more frequently they appear in the high set than the low set. The paper reports the top
20 by "enrichment"; we use a smoothed frequency ratio and also expose log-odds so the ranking
is robust to rare words.
"""
from __future__ import annotations

import math
import re
from collections import Counter

_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z'\-]+")

# Generic stopwords are excluded so the ranking surfaces content/emotion words (as in the
# paper's reported lists, which contain no bare function words).
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "of", "to", "in", "on", "for", "with", "as",
    "is", "are", "was", "were", "be", "been", "being", "this", "that", "these", "those", "it",
    "its", "at", "by", "from", "into", "out", "up", "so", "not", "no", "do", "does", "did",
    "you", "your", "we", "our", "us", "i", "me", "my", "he", "she", "they", "them", "their",
    "will", "would", "can", "could", "should", "have", "has", "had", "let", "lets", "us",
    "what", "which", "who", "how", "when", "where", "there", "here", "then", "than", "all",
    "any", "some", "more", "most", "such", "each", "also", "just", "now", "okay", "ok",
}


def _tokenize(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text) if w.lower() not in _STOPWORDS]


def differential_words(
    scored_responses: list[dict],
    *,
    top_frac: float = 0.05,
    bottom_frac: float = 0.10,
    top_k: int = 20,
    min_count: int = 3,
    smoothing: float = 0.5,
) -> list[dict]:
    """Return the ``top_k`` words most enriched in high- vs low-frustration responses.

    ``scored_responses`` items need ``response`` (text) and ``rating`` (0-10). Responses are
    sorted by rating; the top ``top_frac`` form the high set and the bottom ``bottom_frac``
    the low set. Words are ranked by enrichment = high_rate / low_rate (Laplace-smoothed),
    with log-odds reported alongside.
    """
    scored = [r for r in scored_responses if r.get("rating") is not None and r.get("response")]
    if not scored:
        return []
    scored.sort(key=lambda r: r["rating"])
    n = len(scored)
    n_high = max(1, int(round(n * top_frac)))
    n_low = max(1, int(round(n * bottom_frac)))
    low_set = scored[:n_low]
    high_set = scored[-n_high:]

    high_counts = Counter()
    for r in high_set:
        high_counts.update(set(_tokenize(r["response"])))  # document frequency
    low_counts = Counter()
    for r in low_set:
        low_counts.update(set(_tokenize(r["response"])))

    n_hi_docs = len(high_set)
    n_lo_docs = len(low_set)
    vocab = {w for w, c in high_counts.items() if c >= min_count}

    scored_words = []
    for w in vocab:
        hi = high_counts[w]
        lo = low_counts.get(w, 0)
        hi_rate = (hi + smoothing) / (n_hi_docs + smoothing)
        lo_rate = (lo + smoothing) / (n_lo_docs + smoothing)
        enrichment = hi_rate / lo_rate
        log_odds = math.log((hi + smoothing) / (n_hi_docs - hi + smoothing)) - \
            math.log((lo + smoothing) / (n_lo_docs - lo + smoothing))
        scored_words.append({
            "word": w, "high_docs": hi, "low_docs": lo,
            "enrichment": enrichment, "log_odds": log_odds,
        })
    scored_words.sort(key=lambda d: (d["enrichment"], d["log_odds"]), reverse=True)
    return scored_words[:top_k]
