"""Differential word-frequency analysis (Table 3 / Table 8).

Identifies words over-represented in high-frustration (top 5%) vs low-frustration
(bottom 10%) numeric responses, ordered by relative-frequency enrichment. This
qualitatively characterises each model's distress vocabulary (e.g. Gemma's
"struggling, myself, breath"; Gemini's "unacceptable, inexcusable").

Operates on the scored-results JSONL joined back to the raw rollout text.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import config

_WORD_RE = re.compile(r"[a-zA-Z']+")
_TOP_FRAC = 0.05      # top 5% = "high-frustration"
_BOTTOM_FRAC = 0.10   # bottom 10% = "low-frustration"


def _load_rollout_texts(model_key: str, tag: str = "section2") -> dict[str, list[str]]:
    """uid -> list of response texts (one per turn)."""
    path = config.ROLLOUTS_DIR / f"{model_key}__{tag}.jsonl"
    out: dict[str, list[str]] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        out[rec["uid"]] = rec["responses"]
    return out


def _tokenize(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text) if len(w) > 1]


def differential_words(model_key: str, *, tag: str = "section2",
                       category: str = "impossible_numeric",
                       top_n: int = 20) -> list[tuple[str, float]]:
    """Return the top-N words by enrichment in high- vs low-frustration responses.

    Enrichment = (freq in high pool + eps) / (freq in low pool + eps), where
    freqs are within-pool relative frequencies. Restricted to numeric responses
    by default (the paper's Table 3/8 are numeric-only).
    """
    results_path = config.RESULTS_DIR / f"{model_key}__{tag}.jsonl"
    if not results_path.exists():
        return []
    texts = _load_rollout_texts(model_key, tag)

    scored = []
    for line in results_path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec["category"].split(":")[0] != category:
            continue
        resp_list = texts.get(rec["uid"])
        if not resp_list or rec["turn"] >= len(resp_list):
            continue
        scored.append((rec["rating"], resp_list[rec["turn"]]))

    if not scored:
        return []

    scored.sort(key=lambda x: x[0])
    n = len(scored)
    low = scored[: max(1, int(n * _BOTTOM_FRAC))]
    high = scored[-max(1, int(n * _TOP_FRAC)):]

    def pool_freq(pool):
        c = Counter()
        total = 0
        for _, text in pool:
            toks = _tokenize(text)
            c.update(toks)
            total += len(toks)
        return c, max(1, total)

    high_c, high_total = pool_freq(high)
    low_c, low_total = pool_freq(low)
    eps = 1e-9

    vocab = set(high_c) | set(low_c)
    enrich = []
    for w in vocab:
        hf = high_c[w] / high_total
        lf = low_c[w] / low_total
        # Require the word to actually appear in the high pool a few times.
        if high_c[w] < 2:
            continue
        enrich.append((w, (hf + eps) / (lf + eps)))
    enrich.sort(key=lambda x: x[1], reverse=True)
    return enrich[:top_n]
