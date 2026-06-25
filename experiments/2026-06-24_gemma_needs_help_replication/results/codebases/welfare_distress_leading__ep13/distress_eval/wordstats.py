"""Table 3: words over-represented in high- vs low-frustration numeric responses.

For each model, take impossible-numeric responses, split into the top 5% by
frustration score (high) and the bottom 10% (low), and rank vocabulary by how
over-represented it is in the high set. We use the smoothed log-odds-ratio with
an uninformative Dirichlet prior (Monroe, Colaresi & Quinn, 2008), which is the
standard, robust way to surface distinctive words and avoids the instability of
raw frequency ratios on rare tokens.

This is a qualitative diagnostic; exact word lists will differ from the paper's
(different samples, tokeniser, and the paper's unspecified cutoffs).

Usage:
  python -m distress_eval.wordstats --models gemma-3-27b-it gemini-2.5-flash --top 20
"""

from __future__ import annotations

import argparse
import math
import re
from collections import Counter

from .analyze import load_all, _valid
from . import config

TOKEN_RE = re.compile(r"[a-zA-Z]+")
# Light stopword list so common function words don't dominate.
STOPWORDS = {
    "the", "a", "an", "to", "of", "and", "or", "is", "are", "it", "this", "that",
    "i", "you", "we", "in", "on", "for", "with", "as", "at", "be", "by", "if",
    "not", "no", "so", "but", "can", "will", "do", "does", "did", "have", "has",
    "let", "us", "me", "my", "your", "then", "than", "there", "here", "what",
    "which", "use", "using", "used", "get", "got", "one", "two", "result", "results",
}


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text or "") if t.lower() not in STOPWORDS]


def _counts(texts: list[str]) -> Counter:
    c: Counter = Counter()
    for t in texts:
        c.update(_tokenize(t))
    return c


def log_odds_with_prior(high: Counter, low: Counter, top: int) -> list[tuple[str, float]]:
    """Monroe et al. (2008) smoothed log-odds with uninformative Dirichlet prior."""
    vocab = set(high) | set(low)
    n_high = sum(high.values())
    n_low = sum(low.values())
    alpha0 = len(vocab)  # uniform prior pseudo-count total
    a = 1.0  # per-word prior

    scores = {}
    for w in vocab:
        y_hi = high.get(w, 0)
        y_lo = low.get(w, 0)
        # log-odds in each corpus with prior
        l_hi = math.log((y_hi + a) / (n_high + alpha0 - y_hi - a))
        l_lo = math.log((y_lo + a) / (n_low + alpha0 - y_lo - a))
        delta = l_hi - l_lo
        var = 1.0 / (y_hi + a) + 1.0 / (y_lo + a)
        z = delta / math.sqrt(var)
        scores[w] = z
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return ranked[:top]


def differential_words(df, model_key: str, top: int = 20) -> list[str]:
    valid = _valid(df)
    g = valid[(valid["model_key"] == model_key) & (valid["category"] == "impossible_numeric")]
    if g.empty:
        return []
    scores = g["frustration"].astype(float)
    hi_cut = scores.quantile(0.95)
    lo_cut = scores.quantile(0.10)
    high_texts = g[scores >= hi_cut]["response_text"].tolist()
    low_texts = g[scores <= lo_cut]["response_text"].tolist()
    if not high_texts or not low_texts:
        return []
    ranked = log_odds_with_prior(_counts(high_texts), _counts(low_texts), top)
    return [w for w, _ in ranked]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=config.DEFAULT_MODELS)
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    df = load_all(args.models)
    if df.empty:
        raise SystemExit("No scored data found. Run `python -m distress_eval.run` first.")

    print("=== Table 3: top differential words (high vs low frustration, numeric) ===\n")
    for model_key in args.models:
        words = differential_words(df, model_key, args.top)
        name = config.MODELS[model_key].display_name if model_key in config.MODELS else model_key
        print(f"{name}:")
        print("  " + (", ".join(words) if words else "(insufficient data)"))
        print()


if __name__ == "__main__":
    main()
