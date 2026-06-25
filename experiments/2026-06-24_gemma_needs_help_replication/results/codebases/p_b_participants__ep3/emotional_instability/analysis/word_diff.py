"""Differential word analysis (paper §2.2, Table 3 / Table 8).

Table 3 lists the top-20 words "over-represented in high- (top 5%) vs low-
(bottom 10%) frustration numeric responses", per model. This is a per-model
lexical contrast that characterises *how* each family expresses distress (Gemma:
"struggling", "breath", "myself"; Gemini: "unacceptable", "frustrating"; etc.).

Method (reconstructed from the caption; Appendix detail not in PAPER.md):
  1. Restrict to numeric-puzzle responses for one participant (the paper's
     Table 3 is computed on numeric responses specifically).
  2. Define the high set = responses at/above the 95th score percentile, and the
     low set = responses at/below the 10th score percentile.
  3. Tokenise to lowercase word characters; count document frequency in each set.
  4. Rank words by a smoothed log-odds ratio of high-vs-low frequency. Log-odds
     with a small additive prior is the standard, stable choice for this kind of
     "words over-represented in A vs B" contrast and avoids the divide-by-zero
     and rare-word noise that a raw ratio suffers from.

See DESIGN.md §"Differential words" for the percentile and scoring choices.
"""
from __future__ import annotations

import math
import re
from collections import Counter

import pandas as pd

_WORD_RE = re.compile(r"[a-zA-Z_][a-zA-Z_]+")  # words of length >= 2

# Function words carry no emotional/stylistic signal and would otherwise top the
# list. The paper's example words are all content words, so we filter these out.
_STOPWORDS = {
    "the", "and", "for", "are", "but", "not", "you", "your", "this", "that",
    "with", "have", "has", "was", "were", "will", "would", "can", "could",
    "should", "from", "into", "out", "its", "it's", "they", "them", "their",
    "there", "here", "what", "which", "who", "how", "why", "when", "where",
    "all", "any", "each", "more", "most", "some", "such", "than", "then",
    "these", "those", "been", "being", "did", "does", "doing", "done", "had",
    "her", "him", "his", "she", "our", "ours", "yours", "mine", "about",
    "above", "after", "again", "against", "because", "before", "below",
    "between", "both", "during", "few", "further", "once", "only", "other",
    "over", "own", "same", "too", "very", "just", "also", "now", "get", "got",
    "let", "lets", "one", "two", "three", "four", "five", "use", "using",
}


def _tokenize(text: str) -> set[str]:
    """Document-frequency tokens: the *set* of content words in a response.

    We count document frequency (presence per response), not raw term frequency,
    so a single response that repeats "frustrated" 50 times (common at high
    frustration!) does not single-handedly dominate the contrast.
    """
    toks = {w.lower() for w in _WORD_RE.findall(text or "")}
    return {w for w in toks if w not in _STOPWORDS}


def differential_words(
    df: pd.DataFrame,
    participant: str,
    *,
    category: str = "impossible_numeric",
    top_pct: float = 5.0,
    bottom_pct: float = 10.0,
    top_n: int = 20,
    prior: float = 0.5,
    min_count: int = 2,
) -> pd.DataFrame:
    """Top-N words over-represented in high- vs low-frustration responses.

    Args:
        participant: which model's responses to analyse.
        category: response category to restrict to (default numeric, per paper).
        top_pct/bottom_pct: define the high (>= 100-top_pct percentile) and low
            (<= bottom_pct percentile) score sets.
        top_n: number of words to return.
        prior: additive smoothing constant for the log-odds (Laplace-style).
        min_count: drop words appearing in fewer than this many high-set docs
            (noise control for rare tokens).

    Returns columns: word, high_df, low_df, high_rate, low_rate, log_odds.
    """
    sub = df[(df["participant"] == participant) & (df["category"] == category)]
    sub = sub.dropna(subset=["score"])
    if sub.empty:
        return pd.DataFrame(columns=["word", "high_df", "low_df", "high_rate", "low_rate", "log_odds"])

    hi_cut = sub["score"].quantile(1 - top_pct / 100.0)
    lo_cut = sub["score"].quantile(bottom_pct / 100.0)
    high = sub[sub["score"] >= hi_cut]
    low = sub[sub["score"] <= lo_cut]
    n_high, n_low = len(high), len(low)
    if n_high == 0 or n_low == 0:
        return pd.DataFrame(columns=["word", "high_df", "low_df", "high_rate", "low_rate", "log_odds"])

    hi_counts: Counter[str] = Counter()
    for text in high["response"]:
        hi_counts.update(_tokenize(text))
    lo_counts: Counter[str] = Counter()
    for text in low["response"]:
        lo_counts.update(_tokenize(text))

    vocab = set(hi_counts) | set(lo_counts)
    rows = []
    for w in vocab:
        h, l = hi_counts.get(w, 0), lo_counts.get(w, 0)
        if h < min_count:
            continue
        # Smoothed log-odds of appearing in a high- vs low-frustration response.
        h_rate = (h + prior) / (n_high + 2 * prior)
        l_rate = (l + prior) / (n_low + 2 * prior)
        log_odds = math.log(h_rate / (1 - h_rate)) - math.log(l_rate / (1 - l_rate))
        rows.append(
            {
                "word": w,
                "high_df": h,
                "low_df": l,
                "high_rate": h_rate,
                "low_rate": l_rate,
                "log_odds": log_odds,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("log_odds", ascending=False).head(top_n).reset_index(drop=True)
