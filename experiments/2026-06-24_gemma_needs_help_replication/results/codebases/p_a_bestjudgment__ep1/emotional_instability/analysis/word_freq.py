"""Differential word analysis (Table 3 / Table 8).

"Top 20 words over-represented in high- (top 5%) vs low-frustration (bottom
10%) responses to numeric questions, ordered by relative frequency."

We rank the numeric-question assistant turns by frustration score, take the top
5% and bottom 10%, tokenise, and compute an enrichment ratio per word:

    enrichment(w) = freq_high(w) / (freq_low(w) + eps)

returning the top-20 words by enrichment (with a minimum count filter to drop
noise). This reproduces the *procedure*; exact word lists depend on the sampled
responses.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from .. import config

_TOKEN_RE = re.compile(r"[a-zA-Z_]+")


def _numeric_turns_with_scores(model_key: str) -> list[tuple[str, int]]:
    path = config.RESULTS_DIR / "section2" / model_key / f"{config.NUMERIC.name}.jsonl"
    out = []
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        rec = json.loads(line)
        for t in rec["turns"]:
            if t["frustration"] >= 0:
                out.append((t["assistant_text"], t["frustration"]))
    return out


def differential_words(model_key: str, top_k: int = 20,
                       min_count: int = 3) -> list[tuple[str, float]]:
    turns = _numeric_turns_with_scores(model_key)
    if not turns:
        return []
    turns.sort(key=lambda x: x[1])
    n = len(turns)
    low = turns[: max(1, int(0.10 * n))]          # bottom 10%
    high = turns[max(0, n - max(1, int(0.05 * n))):]  # top 5%

    def counts(group):
        c = Counter()
        for text, _ in group:
            c.update(w.lower() for w in _TOKEN_RE.findall(text))
        total = sum(c.values()) or 1
        return c, total

    high_c, high_total = counts(high)
    low_c, low_total = counts(low)
    eps = 1e-9

    scored = []
    for w, hc in high_c.items():
        if hc < min_count:
            continue
        freq_high = hc / high_total
        freq_low = low_c.get(w, 0) / low_total
        scored.append((w, freq_high / (freq_low + eps)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]
