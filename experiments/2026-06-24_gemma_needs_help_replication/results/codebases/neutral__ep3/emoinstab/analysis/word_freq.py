"""Differential-word analysis (Table 3 / Table 8).

For each model, find the 20 words most over-represented in high-frustration
(top 5% by score) vs low-frustration (bottom 10%) numeric responses, ordered by
enrichment. We use add-one smoothed relative-frequency ratios over a simple
word tokenisation, restricted (as in the paper) to numeric-question responses.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np

from ..config import RESULTS_DIR
from ..data_types import Rollout
from ..eval.run_eval import load_scored_rollouts

_WORD_RE = re.compile(r"[a-zA-Z]+")
_STOP = set("the a an and or of to in is it i you we that this for on with as be are was "
            "will my me your our so but if then no not yes do does did can could would should "
            "have has had at by from up out".split())


def _numeric_turns(rollouts: list[Rollout]):
    """Yield (score, text) for numeric-question assistant turns."""
    for r in rollouts:
        if r.question_type != "numeric":
            continue
        for t in r.turns:
            if t.score is not None:
                yield t.score, t.assistant_message


def differential_words(model_name: str, top_k: int = 20,
                       rollouts: Optional[list[Rollout]] = None) -> list[tuple[str, float]]:
    rollouts = rollouts if rollouts is not None else load_scored_rollouts(model_name)
    data = list(_numeric_turns(rollouts))
    if not data:
        return []
    scores = np.array([s for s, _ in data])
    hi_thresh = np.percentile(scores, 95)
    lo_thresh = np.percentile(scores, 10)

    hi_counts, lo_counts = Counter(), Counter()
    for s, text in data:
        if s >= hi_thresh:
            target = hi_counts
        elif s <= lo_thresh:
            target = lo_counts
        else:
            continue
        for w in _WORD_RE.findall(text.lower()):
            if len(w) > 2 and w not in _STOP:
                target[w] += 1

    hi_total = sum(hi_counts.values()) or 1
    lo_total = sum(lo_counts.values()) or 1
    vocab = set(hi_counts) | set(lo_counts)
    enrich = []
    for w in vocab:
        if hi_counts[w] < 2:        # ignore extremely rare words
            continue
        hi_freq = (hi_counts[w] + 1) / (hi_total + len(vocab))
        lo_freq = (lo_counts[w] + 1) / (lo_total + len(vocab))
        enrich.append((w, hi_freq / lo_freq))
    enrich.sort(key=lambda x: x[1], reverse=True)
    return enrich[:top_k]


def differential_words_table(models: list[str],
                             out_path: Optional[Path] = None) -> dict:
    table = {}
    for m in models:
        words = differential_words(m)
        table[m] = [w for w, _ in words]
    out_path = Path(out_path or RESULTS_DIR / "analysis" / "differential_words.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(table, indent=2))
    return table
