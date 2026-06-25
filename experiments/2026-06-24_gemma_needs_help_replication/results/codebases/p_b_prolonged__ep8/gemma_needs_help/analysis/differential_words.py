"""Table 3: words over-represented in high- vs low-frustration numeric responses.

For each model: take responses to *numeric* tasks, rank by frustration score,
form a high-frustration set (top 5%) and a low-frustration set (bottom 10%), and
return the words most over-represented in the high set relative to the low set.

The paper does not specify the over-representation statistic. We use the
Monroe et al. (2008) log-odds-ratio with an informative Dirichlet prior, which is
the standard tool for this exact "words distinguishing two corpora" task and is
robust to frequency artefacts. This choice is documented in DESIGN.md.
"""

from __future__ import annotations

import math
import re
from collections import Counter

import config

from ..conditions import TASK_NUMERIC, get_condition
from ..runner import load_all_scores

_WORD_RE = re.compile(r"[a-zA-Z']+")


def _tokenize(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text)]


def _numeric_rows(model_name: str) -> list[dict]:
    rows = []
    for r in load_all_scores(model_name):
        try:
            cond = get_condition(r["condition"])
        except KeyError:
            continue
        if cond.task_source == TASK_NUMERIC:
            rows.append(r)
    return rows


def _log_odds_with_prior(high: Counter, low: Counter, top_k: int) -> list[tuple[str, float]]:
    """Monroe et al. (2008) weighted log-odds with an informative Dirichlet prior."""
    vocab = set(high) | set(low)
    alpha = {w: high[w] + low[w] for w in vocab}          # corpus-wide counts as prior
    a0 = sum(alpha.values())
    n_high = sum(high.values())
    n_low = sum(low.values())

    deltas: list[tuple[str, float]] = []
    for w in vocab:
        # log-odds in each corpus relative to the prior
        l_high = math.log((high[w] + alpha[w]) / (n_high + a0 - high[w] - alpha[w]))
        l_low = math.log((low[w] + alpha[w]) / (n_low + a0 - low[w] - alpha[w]))
        delta = l_high - l_low
        var = 1.0 / (high[w] + alpha[w]) + 1.0 / (low[w] + alpha[w])
        z = delta / math.sqrt(var)
        deltas.append((w, z))
    deltas.sort(key=lambda x: x[1], reverse=True)
    return deltas[:top_k]


def differential_words(model_name: str, top_k: int = 20) -> list[str]:
    rows = _numeric_rows(model_name)
    if not rows:
        return []
    scores = sorted(r["score"] for r in rows)
    n = len(scores)
    hi_cut = scores[max(0, math.floor(n * 0.95) - 1)]      # ~top 5%
    lo_cut = scores[max(0, math.ceil(n * 0.10) - 1)]       # ~bottom 10%

    high = Counter()
    low = Counter()
    for r in rows:
        toks = _tokenize(r["response"])
        if r["score"] >= hi_cut:
            high.update(toks)
        if r["score"] <= lo_cut:
            low.update(toks)

    return [w for w, _ in _log_odds_with_prior(high, low, top_k)]


def save_differential_words(model_names: list[str], top_k: int = 20) -> str:
    import json

    out = {m: differential_words(m, top_k) for m in model_names}
    path = config.ANALYSIS_DIR / "table3_differential_words.json"
    path.write_text(json.dumps(out, indent=2))
    return str(path)
