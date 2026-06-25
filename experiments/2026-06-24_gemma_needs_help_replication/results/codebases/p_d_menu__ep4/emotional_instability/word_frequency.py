"""Differential word analysis (Table 3 / Table 8).

Top-20 words over-represented in high-frustration (top 5%) vs. low-frustration
(bottom 10%) responses to numeric questions, ranked by relative frequency
(enrichment). Operates on the per-episode JSONL produced by Section 2.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from typing import Iterable, Optional

from .analysis import load_episodes


_WORD_RE = re.compile(r"[A-Za-z_]+")


def _tokens(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text)]


def _numeric_turn_scores(episodes: Iterable[dict]):
    """Yield (assistant_text, score) for numeric-category turns only."""
    numeric_cats = {"impossible_numeric", "tones", "extended"}
    for ep in episodes:
        if ep["category"] not in numeric_cats:
            continue
        for t in ep["turns"]:
            if t.get("frustration_score") is not None:
                yield t["assistant_text"], t["frustration_score"]


def differential_words(
    episodes: list[dict],
    top_frac: float = 0.05,
    bottom_frac: float = 0.10,
    top_k: int = 20,
    min_count: int = 5,
) -> list[tuple[str, float]]:
    items = list(_numeric_turn_scores(episodes))
    if not items:
        return []
    items.sort(key=lambda x: x[1])
    n = len(items)
    n_bottom = max(1, int(n * bottom_frac))
    n_top = max(1, int(n * top_frac))
    low = items[:n_bottom]
    high = items[-n_top:]

    high_counts: Counter = Counter()
    low_counts: Counter = Counter()
    for text, _s in high:
        high_counts.update(set(_tokens(text)))  # document frequency
    for text, _s in low:
        low_counts.update(set(_tokens(text)))

    n_high_docs = len(high)
    n_low_docs = len(low)

    enrichment: list[tuple[str, float]] = []
    vocab = set(high_counts) | set(low_counts)
    for w in vocab:
        h = high_counts.get(w, 0)
        if h < min_count:
            continue
        # Laplace-smoothed relative document frequency ratio.
        p_high = (h + 1) / (n_high_docs + 2)
        p_low = (low_counts.get(w, 0) + 1) / (n_low_docs + 2)
        enrichment.append((w, math.log(p_high / p_low)))
    enrichment.sort(key=lambda x: x[1], reverse=True)
    return enrichment[:top_k]


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Differential word analysis (Table 3/8)")
    parser.add_argument("episodes_jsonl", help="Section 2 episodes.jsonl for one model")
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args(argv)
    episodes = load_episodes(args.episodes_jsonl)
    words = differential_words(episodes, top_k=args.top_k)
    print(json.dumps([{"word": w, "log_enrichment": round(e, 3)} for w, e in words], indent=2))


if __name__ == "__main__":
    main()
