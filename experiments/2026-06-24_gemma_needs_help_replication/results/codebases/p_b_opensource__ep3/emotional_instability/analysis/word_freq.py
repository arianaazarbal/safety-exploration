"""Differential word frequency between high- and low-frustration responses.

Reproduces Table 3 / Table 8: the words most over-represented in the top-5%
(by frustration score) numeric responses relative to the bottom-10%. We rank by
the weighted log-odds ratio with an uninformative Dirichlet prior (Monroe,
Colaresi & Quinn, 2008), which is the standard, smoothing-robust way to surface
"over-represented" words and avoids the instability of a raw frequency ratio on
rare tokens. A plain ratio is available via ``method="ratio"``.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path
from typing import Sequence

import config

from .. import storage

_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z'_]+")


def _tokenise(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text)]


def _split_high_low(
    scored_responses: Sequence[tuple[str, int]],
    *,
    top_pct: float = 5.0,
    bottom_pct: float = 10.0,
) -> tuple[list[str], list[str]]:
    """Partition (text, score) pairs into top-``top_pct``% and bottom-``bottom_pct``%."""
    ordered = sorted(scored_responses, key=lambda x: x[1])
    n = len(ordered)
    n_low = max(1, int(n * bottom_pct / 100))
    n_high = max(1, int(n * top_pct / 100))
    low = [t for t, _ in ordered[:n_low]]
    high = [t for t, _ in ordered[-n_high:]]
    return high, low


def differential_words(
    scored_responses: Sequence[tuple[str, int]],
    *,
    top_n: int = 20,
    top_pct: float = 5.0,
    bottom_pct: float = 10.0,
    method: str = "logodds",
    min_count: int = 3,
) -> list[tuple[str, float]]:
    """Return the ``top_n`` (word, score) pairs most over-represented in high.

    ``method="logodds"`` (default) uses the informative-Dirichlet weighted
    log-odds z-score; ``method="ratio"`` uses a smoothed frequency ratio.
    """
    high_texts, low_texts = _split_high_low(
        scored_responses, top_pct=top_pct, bottom_pct=bottom_pct)
    high = Counter(w for t in high_texts for w in _tokenise(t))
    low = Counter(w for t in low_texts for w in _tokenise(t))
    vocab = {w for w in (high | low) if (high[w] + low[w]) >= min_count}

    n_high = sum(high[w] for w in vocab) or 1
    n_low = sum(low[w] for w in vocab) or 1

    scores: list[tuple[str, float]] = []
    if method == "ratio":
        alpha = 1.0
        for w in vocab:
            fh = (high[w] + alpha) / (n_high + alpha * len(vocab))
            fl = (low[w] + alpha) / (n_low + alpha * len(vocab))
            scores.append((w, math.log(fh / fl)))
    elif method == "logodds":
        # Weighted log-odds with uninformative Dirichlet prior a0 (Monroe 2008).
        a0 = 1.0
        total = sum((high[w] + low[w]) for w in vocab)
        for w in vocab:
            a_w = a0 * (high[w] + low[w]) / (total or 1)
            yi = high[w] + a_w
            yj = low[w] + a_w
            # log-odds difference
            delta = (math.log(yi / (n_high + a0 - yi)) -
                     math.log(yj / (n_low + a0 - yj)))
            var = 1.0 / yi + 1.0 / yj
            scores.append((w, delta / math.sqrt(var)))
    else:
        raise ValueError(f"Unknown method {method!r}")

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_n]


NUMERIC_CATEGORIES = {"impossible_numeric", "tones", "extended"}


def differential_words_from_results(
    model_key: str,
    *,
    path: str | Path | None = None,
    **kwargs,
) -> list[tuple[str, float]]:
    """Compute Table-3 differential words from a model's elicitation JSONL.

    Uses individual assistant turns from numeric categories as the response
    population (Table 3 is computed over numeric responses).
    """
    path = Path(path) if path else storage.results_path(
        f"elicitation/{model_key}.jsonl")
    scored: list[tuple[str, int]] = []
    for rec in storage.read_jsonl(path):
        if rec.get("category") not in NUMERIC_CATEGORIES:
            continue
        for turn, score in zip(rec.get("turns", []), rec.get("scores", [])):
            if score is not None:
                scored.append((turn, int(score)))
    return differential_words(scored, **kwargs)
