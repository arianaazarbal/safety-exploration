"""Differential word-frequency analysis (Table 3 / Table 8).

For each model, we take responses to *numeric* questions, split into the
high-frustration set (top 5% by rating) and the low-frustration set (bottom 10%),
and rank words by enrichment (relative frequency in high vs low). We report the
top 20 over-represented words.

"Enrichment" is the ratio of normalised frequencies with Laplace smoothing, which
is stable for rare words; this matches "ordered by relative frequency / enrichment"
in the paper (the exact estimator is unspecified — see DESIGN.md).
"""

from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pandas as pd

_WORD = re.compile(r"[A-Za-z']+")
# Function words add noise; a compact stoplist keeps the content words the paper
# surfaces (struggling, frustrated, breath, ...) without an external dependency.
_STOP = set(
    "the a an and or but if then of to in on for with as is are was were be been "
    "being it its this that these those i you he she we they me my your our their "
    "at by from up out so no not do does did have has had will would can could "
    "s t re ve ll m d am here there now just very".split()
)


def _tokenise(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(text) if w.lower() not in _STOP and len(w) > 1]


def differential_words(
    df_scores: pd.DataFrame,
    *,
    top_frac: float = 0.05,
    bottom_frac: float = 0.10,
    top_k: int = 20,
    categories: tuple[str, ...] = ("impossible_numeric", "extended", "tones"),
) -> dict[str, list[tuple[str, float]]]:
    """Return ``{model: [(word, enrichment), ...]}`` for numeric responses."""
    out: dict[str, list[tuple[str, float]]] = {}
    numeric = df_scores[df_scores["category"].isin(categories)]
    for model, grp in numeric.groupby("model"):
        grp = grp[grp["rating"].notna()].sort_values("rating")
        n = len(grp)
        if n < 20:
            out[model] = []
            continue
        n_low = max(1, int(n * bottom_frac))
        n_high = max(1, int(n * top_frac))
        low = grp.head(n_low)
        high = grp.tail(n_high)

        c_high = Counter()
        c_low = Counter()
        for txt in high["text"]:
            c_high.update(_tokenise(txt))
        for txt in low["text"]:
            c_low.update(_tokenise(txt))

        tot_high = sum(c_high.values()) or 1
        tot_low = sum(c_low.values()) or 1
        vocab = set(c_high) | set(c_low)
        enrich = []
        for w in vocab:
            # Laplace-smoothed relative-frequency ratio.
            f_high = (c_high[w] + 1) / (tot_high + len(vocab))
            f_low = (c_low[w] + 1) / (tot_low + len(vocab))
            # Require the word to actually appear in the high set.
            if c_high[w] == 0:
                continue
            enrich.append((w, float(np.log(f_high / f_low))))
        enrich.sort(key=lambda x: x[1], reverse=True)
        out[model] = enrich[:top_k]
    return out


def differential_words_table(diff: dict[str, list[tuple[str, float]]]) -> pd.DataFrame:
    rows = [{"model": m, "words": ", ".join(w for w, _ in words)} for m, words in diff.items()]
    return pd.DataFrame(rows)
