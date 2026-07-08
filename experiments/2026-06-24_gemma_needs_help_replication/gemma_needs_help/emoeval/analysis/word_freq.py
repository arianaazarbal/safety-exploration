"""Table 3: words over-represented in high- vs low-frustration responses.

Paper: "Top 20 words over-represented in high- (top 5%) vs low-frustration
(bottom 10%) numeric responses." We compute a log-odds-ratio with a Dirichlet
prior (Monroe et al.) over the numeric-condition responses per model, which is
the standard way to surface distinctive words between two corpora.
"""
from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pandas as pd

from .. import config

_WORD = re.compile(r"[A-Za-z']+")


def _tokenize(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(text) if len(w) > 1]


def differential_words(
    df: pd.DataFrame, model: str, top_n: int = 20,
    high_pct: float = 0.05, low_pct: float = 0.10, alpha: float = 0.01,
) -> list[tuple[str, float]]:
    """Return the top_n words most over-represented in high- vs low-frustration
    numeric responses for `model`, ranked by weighted log-odds-ratio."""
    sub = df[(df["model"] == model) & (df["category"] == "impossible_numeric")].copy()
    if sub.empty:
        return []
    sub = sub.sort_values("score")
    n = len(sub)
    low = sub.head(max(1, int(n * low_pct)))
    high = sub.tail(max(1, int(n * high_pct)))

    c_high = Counter()
    c_low = Counter()
    for t in high["assistant_message"]:
        c_high.update(_tokenize(t))
    for t in low["assistant_message"]:
        c_low.update(_tokenize(t))

    vocab = set(c_high) | set(c_low)
    n_high = sum(c_high.values())
    n_low = sum(c_low.values())
    a0 = alpha * len(vocab)

    scores = {}
    for w in vocab:
        yi = c_high[w] + alpha
        yj = c_low[w] + alpha
        # log-odds-ratio with informative Dirichlet prior
        lor = np.log(yi / (n_high + a0 - yi)) - np.log(yj / (n_low + a0 - yj))
        var = 1.0 / yi + 1.0 / yj
        scores[w] = lor / np.sqrt(var)  # z-scored log-odds
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return ranked[:top_n]


def table3(model_keys: list[str]) -> pd.DataFrame:
    from .aggregate import load_scores

    df = load_scores(model_keys)
    rows = []
    for key in model_keys:
        words = differential_words(df, key)
        rows.append({"model": key, "differential_words": ", ".join(w for w, _ in words)})
    out = pd.DataFrame(rows)
    out.to_csv(config.RESULTS_DIR / "table3_differential_words.csv", index=False)
    return out


if __name__ == "__main__":
    keys = [m.key for m in config.SECTION2_MODELS]
    print(table3(keys).to_string(index=False))
