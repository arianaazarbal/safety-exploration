"""Differential-word analysis (paper Table 3).

Top words over-represented in high-frustration (top 5%) vs low-frustration
(bottom 10%) numeric responses, per model. We rank words by a smoothed
log-odds-ratio of frequency in the high-frustration set vs the low set, over the
'impossible_numeric' + 'tones' + 'extended' categories (all numeric-puzzle based).
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

from ..config import Config
from ..io_utils import read_jsonl

_WORD = re.compile(r"[A-Za-z_]+")
# Numeric-puzzle-based categories (the paper's Table 3 is over numeric responses).
_NUMERIC_CATEGORIES = {"impossible_numeric", "tones", "extended"}


def _tokens(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(text) if len(w) > 2]


def differential_words(rows: list[dict], top_k: int = 20) -> dict[str, list[str]]:
    by_model: dict[str, list[dict]] = {}
    for r in rows:
        if r.get("category") in _NUMERIC_CATEGORIES and "frustration" in r:
            by_model.setdefault(r["model_key"], []).append(r)

    out: dict[str, list[str]] = {}
    for model, recs in by_model.items():
        recs_sorted = sorted(recs, key=lambda r: r["frustration"])
        n = len(recs_sorted)
        if n < 20:
            out[model] = []
            continue
        low = recs_sorted[: max(1, n // 10)]          # bottom 10%
        high = recs_sorted[-max(1, n // 20):]          # top 5%

        hi_counts = Counter(t for r in high for t in _tokens(r["response"]))
        lo_counts = Counter(t for r in low for t in _tokens(r["response"]))
        hi_total = sum(hi_counts.values()) or 1
        lo_total = sum(lo_counts.values()) or 1

        vocab = set(hi_counts) | set(lo_counts)
        scored = []
        for w in vocab:
            # smoothed log-odds of appearing in high vs low set
            hi_p = (hi_counts[w] + 1) / (hi_total + len(vocab))
            lo_p = (lo_counts[w] + 1) / (lo_total + len(vocab))
            scored.append((math.log(hi_p / lo_p), w))
        scored.sort(reverse=True)
        out[model] = [w for _, w in scored[:top_k]]
    return out


def write_report(cfg: Config) -> dict:
    rows = [r for r in read_jsonl(cfg.paths.scored) if "frustration" in r]
    result = differential_words(rows)
    out_dir = Path(cfg.paths.analysis_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "differential_words.json").write_text(json.dumps(result, indent=2))

    print("\n=== Differential words (high vs low frustration, numeric responses) ===")
    for model, words in result.items():
        print(f"  {model}: {', '.join(words)}")
    return result
