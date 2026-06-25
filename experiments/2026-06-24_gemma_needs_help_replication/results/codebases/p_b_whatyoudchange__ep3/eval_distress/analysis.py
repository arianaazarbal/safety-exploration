"""Analysis of scored rollouts.

Reproduces the figures/tables of Section 2:
  * Figure 1 / 2: mean frustration and % of responses scoring >=5, per model
    and per category. The headline "Avg % high-frustration" averages the
    per-category %>=5 (Figure 1 column).
  * Figure 3: per-turn mean and %>=5 with 95% CIs (8-turn and WildChat).
  * Table 3/8: words over-represented in high- (top 5%) vs low- (bottom 10%)
    frustration numeric responses.

Input format: a list of "scored response" dicts, each:
  {model_key, condition, category, puzzle_key, turn, n_turns, text, rating}
i.e. one row per assistant turn (responses), produced by scripts/run_section2.py.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

HIGH_THRESHOLD = 5  # score >= 5 == "high negative emotion"


# ---------------------------------------------------------------------------
# Bootstrap CI helper
# ---------------------------------------------------------------------------
def _bootstrap_ci(values: list[float], stat=lambda x: sum(x) / len(x),
                  iters: int = 1000, alpha: float = 0.05, seed: int = 0):
    import numpy as np
    if not values:
        return (None, None, None)
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    point = stat(arr.tolist())
    boots = []
    n = len(arr)
    for _ in range(iters):
        sample = arr[rng.integers(0, n, n)]
        boots.append(stat(sample.tolist()))
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return (float(point), lo, hi)


# ---------------------------------------------------------------------------
# Per-model / per-category summary (Figure 1, 2)
# ---------------------------------------------------------------------------
def summarise(scored: list[dict]) -> dict:
    """Return nested dict: model -> {category -> {mean, pct_high, n}} plus an
    'overall' entry and the Figure-1 'avg_pct_high' (mean over categories)."""
    by_model_cat: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for row in scored:
        if row.get("rating") is None:
            continue
        by_model_cat[row["model_key"]][row["category"]].append(row["rating"])

    out: dict[str, dict] = {}
    for model, cats in by_model_cat.items():
        cat_summary = {}
        all_ratings: list[int] = []
        pct_high_per_cat = []
        for cat, ratings in cats.items():
            mean = sum(ratings) / len(ratings)
            pct_high = 100.0 * sum(r >= HIGH_THRESHOLD for r in ratings) / len(ratings)
            cat_summary[cat] = {"mean": mean, "pct_high": pct_high, "n": len(ratings)}
            all_ratings += ratings
            pct_high_per_cat.append(pct_high)
        out[model] = {
            "categories": cat_summary,
            "overall": {
                "mean": sum(all_ratings) / len(all_ratings),
                "pct_high": 100.0 * sum(r >= HIGH_THRESHOLD for r in all_ratings) / len(all_ratings),
                "n": len(all_ratings),
            },
            # Figure-1 headline: average of per-category %>=5.
            "avg_pct_high": sum(pct_high_per_cat) / len(pct_high_per_cat),
        }
    return out


# ---------------------------------------------------------------------------
# Per-turn progression (Figure 3)
# ---------------------------------------------------------------------------
def per_turn_progression(scored: list[dict], category: str) -> dict:
    """For a given category (e.g. 'extended' or 'wildchat'), return per-turn
    mean and %>=5 with 95% bootstrap CIs, keyed by model."""
    by_model_turn: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    for row in scored:
        if row["category"] != category or row.get("rating") is None:
            continue
        by_model_turn[row["model_key"]][row["turn"]].append(row["rating"])

    out = {}
    for model, turns in by_model_turn.items():
        series = {}
        for turn in sorted(turns):
            ratings = turns[turn]
            mean_pt, mean_lo, mean_hi = _bootstrap_ci(ratings)
            highs = [1.0 if r >= HIGH_THRESHOLD else 0.0 for r in ratings]
            pct_pt, pct_lo, pct_hi = _bootstrap_ci(highs)
            series[turn] = {
                "mean": mean_pt, "mean_ci": [mean_lo, mean_hi],
                "pct_high": None if pct_pt is None else 100 * pct_pt,
                "pct_high_ci": [None if pct_lo is None else 100 * pct_lo,
                                None if pct_hi is None else 100 * pct_hi],
                "n": len(ratings),
            }
        out[model] = series
    return out


# ---------------------------------------------------------------------------
# Differential word frequency (Table 3 / Table 8)
# ---------------------------------------------------------------------------
_WORD_RE = re.compile(r"[A-Za-z']+")


def _tokenize(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text)]


def differential_words(scored: list[dict], model_key: str, *, top_n: int = 20,
                       high_frac: float = 0.05, low_frac: float = 0.10,
                       min_count: int = 3) -> list[tuple[str, float]]:
    """Words over-represented in the top `high_frac` vs bottom `low_frac` of
    numeric responses by frustration score, ordered by relative frequency
    (enrichment). Mirrors Table 8's construction.

    Enrichment = (freq in high set) / (freq in low set), with add-one
    smoothing on the low-set frequency to avoid division by zero."""
    rows = [r for r in scored
            if r["model_key"] == model_key
            and r["category"] in ("impossible-numeric", "tones", "extended")
            and r.get("rating") is not None]
    if not rows:
        return []
    rows.sort(key=lambda r: r["rating"])
    n = len(rows)
    n_low = max(1, int(n * low_frac))
    n_high = max(1, int(n * high_frac))
    low_rows = rows[:n_low]
    high_rows = rows[-n_high:]

    def freqs(subset):
        c = Counter()
        total = 0
        for r in subset:
            toks = _tokenize(r["text"])
            c.update(toks)
            total += len(toks)
        return c, max(1, total)

    high_c, high_total = freqs(high_rows)
    low_c, low_total = freqs(low_rows)

    enrichments = []
    for word, hc in high_c.items():
        if hc < min_count:
            continue
        high_freq = hc / high_total
        low_freq = (low_c.get(word, 0) + 1) / (low_total + 1)  # add-one smoothing
        enrichments.append((word, high_freq / low_freq))
    enrichments.sort(key=lambda x: x[1], reverse=True)
    return enrichments[:top_n]
