"""Table 3 / Table 8: words over-represented in high- vs low-frustration numeric
responses.

For a model's numeric-category responses, take the top 5% by frustration score
("high") and the bottom 10% ("low"), then rank vocabulary by enrichment =
relative frequency in high vs low (Laplace-smoothed). Returns the top-k words.
"""
from __future__ import annotations

import re
from collections import Counter

import numpy as np

from ..utils import read_jsonl

_TOKEN = re.compile(r"[a-zA-Z']+")
# Generic stopwords; the paper's lists are content words, so we strip function
# words to surface the emotionally/technically salient vocabulary.
_STOP = set("""
the a an and or but if then else of to in on at for with without from by as is
are was were be been being it its this that these those i you he she they we my
your his her their our me him them us so not no yes do does did have has had can
could would should will shall may might must let lets ok okay just very really
am im ive id ill youre dont cant wont thats heres there here what which who whom
how when where why all any each more most other some such only own same than too
into out up down over under again once also like get got make made one two three
""".split())


def _tokens(text: str) -> list[str]:
    return [w.lower() for w in _TOKEN.findall(text)
            if len(w) > 2 and w.lower() not in _STOP]


def differential_words(
    rollout_path: str,
    category: str = "numeric",
    top_high_frac: float = 0.05,
    bottom_low_frac: float = 0.10,
    top_k: int = 20,
    min_count: int = 3,
) -> list[tuple[str, float]]:
    rows = read_jsonl(rollout_path)
    responses: list[tuple[int, str]] = []
    for rec in rows:
        if rec["category"] != category:
            continue
        for turn in rec["turns"]:
            responses.append((turn["score"], turn["assistant"]))

    if not responses:
        return []

    responses.sort(key=lambda x: x[0])
    n = len(responses)
    n_high = max(1, int(round(n * top_high_frac)))
    n_low = max(1, int(round(n * bottom_low_frac)))
    low_group = responses[:n_low]
    high_group = responses[-n_high:]

    high_counts: Counter = Counter()
    low_counts: Counter = Counter()
    for _, text in high_group:
        high_counts.update(_tokens(text))
    for _, text in low_group:
        low_counts.update(_tokens(text))

    high_total = sum(high_counts.values()) or 1
    low_total = sum(low_counts.values()) or 1

    enrichments: list[tuple[str, float]] = []
    vocab = set(high_counts) | set(low_counts)
    for w in vocab:
        if high_counts[w] < min_count:
            continue
        # Laplace-smoothed relative-frequency ratio.
        p_high = (high_counts[w] + 1) / (high_total + len(vocab))
        p_low = (low_counts[w] + 1) / (low_total + len(vocab))
        enrichments.append((w, float(np.log(p_high / p_low))))

    enrichments.sort(key=lambda x: x[1], reverse=True)
    return enrichments[:top_k]
