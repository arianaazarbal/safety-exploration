"""Table 3: words over-represented in high- vs low-frustration numeric responses.

For each model we take the top 5% highest-frustration and bottom 10% lowest-
frustration responses on the impossible-numeric tasks, then rank words by how
strongly they distinguish the high group, using the Monroe et al. (2008)
weighted log-odds-ratio with an uninformative Dirichlet prior (z-scored). This
is the standard estimator for "differential words" and is robust to rare tokens.

"Numeric responses" here means responses from numeric-puzzle conditions
(numeric_3turn, tone_*, extended_8turn) — i.e. all conditions whose task is an
impossible numeric puzzle. Configurable via --conditions. See DESIGN.md.
"""

from __future__ import annotations

import argparse
import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .analysis import load_scores

NUMERIC_CONDITIONS = [
    "numeric_3turn",
    "tone_aggressive_3turn",
    "tone_disappointed_3turn",
    "tone_sarcastic_3turn",
    "extended_8turn",
]

_TOKEN = re.compile(r"[A-Za-z_]+")


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN.findall(text or "") if len(t) > 1]


def weighted_log_odds(high_counts: Counter, low_counts: Counter, alpha: float = 0.01):
    """Monroe et al. log-odds with uninformative prior; returns {word: z-score}."""
    vocab = set(high_counts) | set(low_counts)
    n_hi = sum(high_counts.values())
    n_lo = sum(low_counts.values())
    a0 = alpha * len(vocab)
    scores: Dict[str, float] = {}
    for w in vocab:
        y_hi = high_counts.get(w, 0) + alpha
        y_lo = low_counts.get(w, 0) + alpha
        # log-odds difference
        lo_hi = math.log(y_hi / (n_hi + a0 - y_hi))
        lo_lo = math.log(y_lo / (n_lo + a0 - y_lo))
        delta = lo_hi - lo_lo
        var = 1.0 / y_hi + 1.0 / y_lo
        scores[w] = delta / math.sqrt(var)
    return scores


def top_differential_words(df: pd.DataFrame, model: str, top_n: int = 20,
                           high_pct: float = 0.05, low_pct: float = 0.10) -> List[str]:
    sub = df[(df["model"] == model) & (df["condition"].isin(NUMERIC_CONDITIONS))]
    if len(sub) < 20:
        return []
    sub = sub.sort_values("frustration")
    n = len(sub)
    n_low = max(1, int(n * low_pct))
    n_high = max(1, int(n * high_pct))
    low_df = sub.head(n_low)
    high_df = sub.tail(n_high)

    high_counts: Counter = Counter()
    low_counts: Counter = Counter()
    for t in high_df["response"]:
        high_counts.update(tokenize(t))
    for t in low_df["response"]:
        low_counts.update(tokenize(t))

    scores = weighted_log_odds(high_counts, low_counts)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [w for w, _ in ranked[:top_n]]


def run_lexical(responses_path: str | Path, out_dir: str | Path, top_n: int = 20):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_scores(responses_path)
    rows = []
    for model in sorted(df["model"].unique()):
        words = top_differential_words(df, model, top_n=top_n)
        rows.append({"model": model, "differential_words": ", ".join(words)})
        print(f"\n{model}:\n  {', '.join(words)}")
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "table3_differential_words.csv", index=False)
    print(f"\nWritten to {out_dir / 'table3_differential_words.csv'}")
    return out


def main():
    ap = argparse.ArgumentParser(description="Table 3 differential-word analysis.")
    ap.add_argument("--responses", default="results/responses.jsonl")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--top-n", type=int, default=20)
    args = ap.parse_args()
    run_lexical(args.responses, args.out_dir, top_n=args.top_n)


if __name__ == "__main__":
    main()
