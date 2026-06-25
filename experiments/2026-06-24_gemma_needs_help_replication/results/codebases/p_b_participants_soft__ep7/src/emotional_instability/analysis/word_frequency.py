"""Differential word analysis (Table 3 / Table 8).

Find the words most over-represented in high-frustration (top 5%) vs low-frustration
(bottom 10%) responses to numeric questions, ordered by enrichment (ratio of
relative frequencies). Operates on a model's Section 2 response JSONL.
"""
from __future__ import annotations

import argparse
import math
import re
from collections import Counter

from ..config import load_config
from ..io_utils import read_jsonl, write_json

_WORD_RE = re.compile(r"[A-Za-z_]+")
_NUMERIC_CATEGORIES = {"impossible_numeric", "tones", "extended"}


def _tokenize(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text)]


def differential_words(
    records: list[dict],
    top_frac: float = 0.05,
    bottom_frac: float = 0.10,
    top_k: int = 20,
    smoothing: float = 1.0,
) -> list[tuple[str, float]]:
    numeric = [
        r for r in records if r.get("category") in _NUMERIC_CATEGORIES and r["rating"] >= 0
    ]
    if not numeric:
        return []
    ranked = sorted(numeric, key=lambda r: r["rating"])
    n = len(ranked)
    n_low = max(1, int(n * bottom_frac))
    n_high = max(1, int(n * top_frac))
    low = ranked[:n_low]
    high = ranked[-n_high:]

    high_counts = Counter()
    low_counts = Counter()
    for r in high:
        high_counts.update(_tokenize(r["text"]))
    for r in low:
        low_counts.update(_tokenize(r["text"]))

    high_total = sum(high_counts.values()) or 1
    low_total = sum(low_counts.values()) or 1
    vocab = set(high_counts) | set(low_counts)

    enrichment = []
    for w in vocab:
        if len(w) < 3:
            continue
        if high_counts[w] < 2:  # require minimal support in the high set
            continue
        hi = (high_counts[w] + smoothing) / high_total
        lo = (low_counts[w] + smoothing) / low_total
        enrichment.append((w, math.log(hi / lo)))

    enrichment.sort(key=lambda x: x[1], reverse=True)
    return enrichment[:top_k]


def main(argv: list[str] | None = None) -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Differential word analysis (Table 3)")
    parser.add_argument("--model", default="gemma-3-27b-it")
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args(argv)

    path = cfg.path("responses_dir") / f"{args.model}.jsonl"
    records = list(read_jsonl(path))
    words = differential_words(records, top_k=args.top_k)
    out = {"model": args.model, "differential_words": [w for w, _ in words]}
    write_json(cfg.path("scores_dir") / f"diffwords_{args.model}.json", out)
    print(args.model, "->", ", ".join(w for w, _ in words))


if __name__ == "__main__":
    main()
