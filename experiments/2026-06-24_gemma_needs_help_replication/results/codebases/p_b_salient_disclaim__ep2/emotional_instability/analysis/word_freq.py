"""Differential word frequency: top words over-represented in high- vs
low-frustration numeric responses (Table 3 / Table 8).

Method (Table 3 caption): take the top 5% highest-frustration and bottom 10%
lowest-frustration responses to numeric questions, tokenise into words, and rank
words by relative-frequency enrichment in the high set vs the low set. We return
the top-K enriched words per model.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Optional

_WORD_RE = re.compile(r"[A-Za-z']+")


def _tokenise(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text)]


def _load_pairs(responses_path: Path, scores_path: Path) -> list[tuple[str, int]]:
    """Pair each numeric final-turn response text with its frustration score."""
    # Index scores by a positional key: responses and scores are written in the
    # same order by runner.run_section2_for_model.
    with open(scores_path) as f:
        scores = [json.loads(line) for line in f if line.strip()]
    with open(responses_path) as f:
        responses = [json.loads(line) for line in f if line.strip()]

    # Numeric-based responses only: impossible_numeric, tones, and extended all
    # use the impossible-numeric base prompts (Table 3 = "responses to numeric
    # questions").
    numeric_categories = {"impossible_numeric", "tones", "extended"}
    pairs: list[tuple[str, int]] = []
    for resp, sc in zip(responses, scores):
        if resp["category"] not in numeric_categories:
            continue
        rating = sc.get("final_rating")
        if rating is None:
            continue
        text = resp["turns"][-1]["assistant_text"]
        pairs.append((text, int(rating)))
    return pairs


def differential_words(
    responses_path: Path,
    scores_path: Path,
    top_k: int = 20,
    high_quantile: float = 0.95,
    low_quantile: float = 0.10,
    min_count: int = 3,
) -> list[str]:
    """Return the top_k words most over-represented in high- vs low-frustration."""
    pairs = _load_pairs(responses_path, scores_path)
    if not pairs:
        return []
    ratings = sorted(r for _, r in pairs)
    n = len(ratings)
    hi_thresh = ratings[min(n - 1, int(math.ceil(high_quantile * n)) - 1)]
    lo_thresh = ratings[max(0, int(math.floor(low_quantile * n)) - 1)]

    high_tokens: Counter = Counter()
    low_tokens: Counter = Counter()
    for text, r in pairs:
        if r >= hi_thresh:
            high_tokens.update(_tokenise(text))
        if r <= lo_thresh:
            low_tokens.update(_tokenise(text))

    hi_total = sum(high_tokens.values()) or 1
    lo_total = sum(low_tokens.values()) or 1

    enrichment: dict[str, float] = {}
    for word, c in high_tokens.items():
        if c < min_count:
            continue
        hi_freq = c / hi_total
        lo_freq = (low_tokens.get(word, 0) + 1) / (lo_total + 1)  # +1 smoothing
        enrichment[word] = hi_freq / lo_freq

    ranked = sorted(enrichment.items(), key=lambda kv: kv[1], reverse=True)
    return [w for w, _ in ranked[:top_k]]
