"""Differential word frequency (Table 3 / Table 8).

For each model's numeric-puzzle responses, rank by frustration score, take the
top 5% (high) and bottom 10% (low), and report the 20 words most enriched in
high vs low responses by relative frequency.
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import numpy as np

from gnh.io import read_jsonl

_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z'_]+")
_NUMERIC_CATEGORIES = {"impossible_numeric", "tones", "extended"}


def _tokenize(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text)]


def differential_words(
    gen_store_path: str | Path,
    judge_store_path: str | Path,
    *,
    top_frac: float = 0.05,
    bottom_frac: float = 0.10,
    top_k: int = 20,
    min_count: int = 3,
    smoothing: float = 1.0,
) -> dict[str, list[tuple[str, float]]]:
    gen_by_key = {r["key"]: r for r in read_jsonl(gen_store_path)}
    # model -> list of (score, text) for numeric responses
    per_model: dict[str, list[tuple[int, str]]] = {}
    for j in read_jsonl(judge_store_path):
        if j.get("score") is None:
            continue
        g = gen_by_key.get(j["gen_key"])
        if not g or g["category"] not in _NUMERIC_CATEGORIES:
            continue
        text = g["turns"][j["turn_index"]]["assistant"]
        per_model.setdefault(j["model"], []).append((int(j["score"]), text))

    out: dict[str, list[tuple[str, float]]] = {}
    for model, items in per_model.items():
        if len(items) < 20:
            out[model] = []
            continue
        items.sort(key=lambda x: x[0])
        n = len(items)
        low = items[: max(1, int(n * bottom_frac))]
        high = items[-max(1, int(n * top_frac)) :]
        high_counts = Counter()
        low_counts = Counter()
        for _, t in high:
            high_counts.update(_tokenize(t))
        for _, t in low:
            low_counts.update(_tokenize(t))
        high_total = sum(high_counts.values()) or 1
        low_total = sum(low_counts.values()) or 1

        scores: list[tuple[str, float]] = []
        for word, hc in high_counts.items():
            if hc < min_count:
                continue
            hf = hc / high_total
            lf = (low_counts.get(word, 0) + smoothing) / (low_total + smoothing)
            enrichment = float(np.log(hf / lf))
            scores.append((word, enrichment))
        scores.sort(key=lambda x: x[1], reverse=True)
        out[model] = scores[:top_k]
    return out
