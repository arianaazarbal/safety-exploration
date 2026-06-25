"""Table 3 / 8: words over-represented in high- vs low-frustration responses.

For each model, take numeric-question responses, split into the top-5% (high)
and bottom-10% (low) by frustration score, and rank words by relative frequency
(enrichment) in high vs low. Returns the top-N differential words.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from ..config import RESULTS_DIR

_NUMERIC_CATEGORIES = ("numeric", "tones", "extended")
_TOKEN_RE = re.compile(r"[A-Za-z_]+")


def _numeric_responses(model_key: str) -> list[tuple[int, str]]:
    model_dir = RESULTS_DIR / model_key / "distress"
    out: list[tuple[int, str]] = []
    for cat in _NUMERIC_CATEGORIES:
        path = model_dir / f"{cat}.jsonl"
        if not path.exists():
            continue
        for line in path.open():
            conv = json.loads(line)
            for t in conv["turns"]:
                if t.get("score") is not None:
                    out.append((t["score"], t["assistant_response"]))
    return out


def _word_counts(texts: list[str]) -> Counter:
    c = Counter()
    for text in texts:
        c.update(w.lower() for w in _TOKEN_RE.findall(text))
    return c


def differential_words(model_key: str, top_n: int = 20,
                       high_pct: float = 0.05, low_pct: float = 0.10,
                       laplace: float = 1.0) -> list[str]:
    responses = _numeric_responses(model_key)
    if not responses:
        return []
    responses.sort(key=lambda x: x[0])
    n = len(responses)
    n_high = max(1, int(n * high_pct))
    n_low = max(1, int(n * low_pct))
    high_texts = [t for _, t in responses[-n_high:]]
    low_texts = [t for _, t in responses[:n_low]]

    high_c = _word_counts(high_texts)
    low_c = _word_counts(low_texts)
    high_total = sum(high_c.values()) or 1
    low_total = sum(low_c.values()) or 1

    enrichment = {}
    for word, hc in high_c.items():
        if hc < 2:                      # ignore one-offs
            continue
        hf = hc / high_total
        lf = (low_c.get(word, 0) + laplace) / (low_total + laplace)
        enrichment[word] = hf / lf
    ranked = sorted(enrichment, key=enrichment.get, reverse=True)
    return ranked[:top_n]
