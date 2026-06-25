"""Differential word frequency in frustrated responses (Table 3 / Table 8).

Top-N words over-represented in high-frustration (top 5%) vs low-frustration
(bottom 10%) responses to numeric questions, ordered by enrichment. We compute a
smoothed relative-frequency ratio so rare-but-exclusive words don't dominate.
"""
from __future__ import annotations

import re
from collections import Counter

import pandas as pd

_TOKEN_RE = re.compile(r"[a-zA-Z']+")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text) if len(t) > 1]


def differential_words(
    scores_path,
    category: str = "numeric",
    top_frac: float = 0.05,
    bottom_frac: float = 0.10,
    n: int = 20,
    smoothing: float = 1.0,
) -> list[tuple[str, float]]:
    df = pd.read_json(scores_path, lines=True)
    df = df[df["category"] == category]
    if df.empty:
        return []
    df = df.sort_values("rating")
    n_total = len(df)
    n_top = max(1, int(round(top_frac * n_total)))
    n_bottom = max(1, int(round(bottom_frac * n_total)))

    high = df.tail(n_top)["response"].tolist()
    low = df.head(n_bottom)["response"].tolist()

    high_counts = Counter(t for r in high for t in _tokenize(r))
    low_counts = Counter(t for r in low for t in _tokenize(r))
    high_total = sum(high_counts.values()) or 1
    low_total = sum(low_counts.values()) or 1

    enrichment = {}
    for word, hc in high_counts.items():
        hf = hc / high_total
        lf = (low_counts.get(word, 0) + smoothing / low_total) / low_total
        enrichment[word] = hf / lf
    ranked = sorted(enrichment.items(), key=lambda kv: kv[1], reverse=True)
    return ranked[:n]


def run_word_frequency(cfg, model_label: str, n: int = 20) -> list[tuple[str, float]]:
    import json

    scores_path = cfg.results_dir / "elicitation" / model_label.replace("/", "_") / "scores.jsonl"
    words = differential_words(scores_path, n=n)
    out_dir = cfg.results_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"words_{model_label.replace('/', '_')}.json", "w") as f:
        json.dump(words, f, indent=2)
    print(f"[words:{model_label}] {[w for w, _ in words]}")
    return words
