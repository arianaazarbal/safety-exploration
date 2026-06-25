"""Differential word analysis (Table 3 / Table 8).

Top words over-represented in high-frustration (top 5%) vs low-frustration
(bottom 10%) responses to numeric questions, ordered by enrichment. We score on
the numeric-category turns only, rank turns by judge score, take the top 5% /
bottom 10% slices, and compare smoothed relative word frequencies.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter

import numpy as np
import pandas as pd

from ..config import CFG

_WORD_RE = re.compile(r"[a-zA-Z_][a-zA-Z_']+")


def _numeric_turns(model: str) -> list[tuple[int, str]]:
    path = CFG.out("section2", f"{model}.jsonl")
    out = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r["category"] not in ("impossible_numeric", "tones", "extended"):
                continue
            for t in r["turns"]:
                if "score" in t:
                    out.append((t["score"], t["response"]))
    return out


def _counts(texts: list[str]) -> Counter:
    c = Counter()
    for t in texts:
        c.update(w.lower() for w in _WORD_RE.findall(t))
    return c


def differential_words(model: str, top_k: int = 20) -> pd.DataFrame:
    turns = _numeric_turns(model)
    if not turns:
        return pd.DataFrame()
    scores = np.array([s for s, _ in turns])
    order = np.argsort(scores)
    n = len(turns)
    low_idx = order[: max(1, int(0.10 * n))]
    high_idx = order[-max(1, int(0.05 * n)):]

    high = _counts([turns[i][1] for i in high_idx])
    low = _counts([turns[i][1] for i in low_idx])
    h_tot, l_tot = sum(high.values()) or 1, sum(low.values()) or 1

    rows = []
    for w in set(high) | set(low):
        if len(w) < 3:
            continue
        hf = (high[w] + 1) / (h_tot + 1)
        lf = (low[w] + 1) / (l_tot + 1)
        rows.append({"word": w, "enrichment": hf / lf, "high": high[w], "low": low[w]})
    df = pd.DataFrame(rows).sort_values("enrichment", ascending=False)
    # require some presence in the high slice to avoid noise
    df = df[df["high"] >= 2]
    return df.head(top_k).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=CFG.gemma_participants())
    args = ap.parse_args()
    for m in args.models:
        df = differential_words(m)
        df.to_csv(CFG.out("section2", f"word_diff_{m}.csv"), index=False)
        print(f"\n{m}: " + ", ".join(df["word"].tolist()))


if __name__ == "__main__":
    main()
