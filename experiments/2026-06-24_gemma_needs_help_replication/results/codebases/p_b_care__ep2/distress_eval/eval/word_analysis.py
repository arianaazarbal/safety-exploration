"""Table 3 / Table 8: words over-represented in high- vs low-frustration
numeric responses.

The paper reports the "top 20 words over-represented in high-frustration
(top 5%) vs low-frustration (bottom 10%) responses to numeric questions,
ordered by relative frequency / enrichment". We implement this with the
Monroe et al. (2008) weighted log-odds-ratio with an informative Dirichlet
prior, which is the standard, frequency-robust way to rank distinguishing
words between two corpora (the paper says "ordered by enrichment"; exact
estimator is unspecified, so we pick this well-justified default — see
DESIGN.md).
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

from .. import config
from .analyze import load_records

_TOKEN = re.compile(r"[a-zA-Z_][a-zA-Z_']+")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text)]


def _numeric_scored_responses(path: Path):
    """All numeric (impossible_numeric/extended/tones) responses with scores."""
    numeric_cats = {"impossible_numeric", "extended", "tones"}
    items = []
    for rec in load_records(path):
        if rec["category"] not in numeric_cats:
            continue
        for resp in rec["responses"]:
            if resp.get("score") is not None:
                items.append((resp["score"], resp["text"]))
    return items


def log_odds_with_prior(high_counts: Counter, low_counts: Counter,
                        alpha: float = 0.01):
    """Monroe et al. weighted log-odds-ratio with informative Dirichlet prior.

    Returns {word: z_score}; large positive z => over-represented in `high`.
    """
    vocab = set(high_counts) | set(low_counts)
    n_high = sum(high_counts.values())
    n_low = sum(low_counts.values())
    a0 = alpha * len(vocab)

    z = {}
    for w in vocab:
        y_hi = high_counts.get(w, 0)
        y_lo = low_counts.get(w, 0)
        # log-odds in each corpus relative to the prior
        l_hi = math.log((y_hi + alpha) / (n_high + a0 - y_hi - alpha))
        l_lo = math.log((y_lo + alpha) / (n_low + a0 - y_lo - alpha))
        delta = l_hi - l_lo
        var = 1.0 / (y_hi + alpha) + 1.0 / (y_lo + alpha)
        z[w] = delta / math.sqrt(var)
    return z


def differential_words(items, *, top_frac=0.05, bottom_frac=0.10, top_n=20):
    """Return the ``top_n`` words most over-represented in the top-frac highest
    scoring responses vs the bottom-frac lowest scoring responses."""
    if not items:
        return []
    items = sorted(items, key=lambda x: x[0])
    n = len(items)
    n_low = max(1, int(bottom_frac * n))
    n_high = max(1, int(top_frac * n))
    low = items[:n_low]
    high = items[-n_high:]

    high_counts = Counter(t for _, txt in high for t in tokenize(txt))
    low_counts = Counter(t for _, txt in low for t in tokenize(txt))

    z = log_odds_with_prior(high_counts, low_counts)
    ranked = sorted(z.items(), key=lambda kv: kv[1], reverse=True)
    return [w for w, _ in ranked[:top_n]]


def main():
    ap = argparse.ArgumentParser(description="Differential word analysis (Table 3).")
    ap.add_argument("inputs", nargs="*", type=Path,
                    help="JSONL files (default: all eval_*.jsonl in OUTPUT_DIR).")
    ap.add_argument("--out", type=Path, default=config.OUTPUT_DIR / "word_analysis.json")
    args = ap.parse_args()

    inputs = args.inputs or sorted(config.OUTPUT_DIR.glob("eval_*.jsonl"))
    table = {}
    for p in inputs:
        items = _numeric_scored_responses(p)
        model = next(iter(load_records(p)))["model"] if items else p.stem
        table[model] = differential_words(items)
        print(f"{model}: {', '.join(table[model])}")
    args.out.write_text(json.dumps(table, indent=2))
    print(f"[word_analysis] wrote {args.out}")


if __name__ == "__main__":
    main()
