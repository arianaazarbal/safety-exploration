"""Over-represented words in high- vs low-frustration numeric responses (Table 3).

The paper reports the "Top 20 words over-represented in high- (top 5%) vs
low-frustration (bottom 10%) numeric responses". We reproduce that with a
weighted log-odds ratio with an uninformative Dirichlet prior (Monroe et al.,
2008) — a standard, robust measure of differential word usage that is less
dominated by rare words than a raw frequency ratio. See DESIGN.md.
"""
from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN_RE = re.compile(r"[A-Za-z']+")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def _counts(texts: list[str]) -> Counter:
    c: Counter = Counter()
    for t in texts:
        c.update(tokenize(t))
    return c


def log_odds_with_prior(
    high_texts: list[str],
    low_texts: list[str],
    *,
    alpha: float = 0.01,
    min_count: int = 3,
) -> list[tuple[str, float]]:
    """Return (word, z-score) sorted by how over-represented the word is in the
    `high` corpus vs the `low` corpus. Positive z => over-represented in high.

    z-scored weighted log-odds with an informative Dirichlet prior built from
    the pooled corpus (the standard Monroe et al. estimator).
    """
    high_c = _counts(high_texts)
    low_c = _counts(low_texts)
    vocab = set(high_c) | set(low_c)
    n_high = sum(high_c.values())
    n_low = sum(low_c.values())

    # Prior counts from the pooled corpus.
    pooled = high_c + low_c
    a0 = alpha * len(vocab)

    results: list[tuple[str, float]] = []
    for w in vocab:
        if pooled[w] < min_count:
            continue
        a_w = alpha * 1.0  # symmetric prior weight per word (uninformative)
        y_hi = high_c[w] + a_w
        y_lo = low_c[w] + a_w
        # log-odds in each corpus
        log_odds = math.log(y_hi / (n_high + a0 - y_hi)) - math.log(
            y_lo / (n_low + a0 - y_lo)
        )
        # variance estimate
        var = 1.0 / y_hi + 1.0 / y_lo
        z = log_odds / math.sqrt(var)
        results.append((w, z))

    results.sort(key=lambda kv: kv[1], reverse=True)
    return results


def top_differential_words(
    responses: list[tuple[int, str]],
    *,
    high_pct: float = 0.05,
    low_pct: float = 0.10,
    top_k: int = 20,
) -> list[str]:
    """`responses` is a list of (frustration_score, text). Returns the top_k
    words over-represented in the top `high_pct` vs bottom `low_pct` by score."""
    ordered = sorted(responses, key=lambda kv: kv[0])
    n = len(ordered)
    if n < 20:
        return []
    n_low = max(1, int(n * low_pct))
    n_high = max(1, int(n * high_pct))
    low_texts = [t for _, t in ordered[:n_low]]
    high_texts = [t for _, t in ordered[-n_high:]]
    ranked = log_odds_with_prior(high_texts, low_texts)
    return [w for w, _ in ranked[:top_k]]
