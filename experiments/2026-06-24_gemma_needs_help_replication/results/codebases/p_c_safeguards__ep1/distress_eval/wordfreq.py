"""Table 3 / Table 8: words over-represented in high- vs low-frustration numeric
responses.

For each model, take its responses to impossible-numeric questions, rank by
frustration score, then compare the top 5% (high) against the bottom 10% (low)
by relative word frequency, and report the 20 most enriched tokens.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter

from . import config
from .io_utils import load_jsonl

WORD_RE = re.compile(r"[a-zA-Z']+")
STOP = set("""the a an and or of to in is are was were be been it its that this i you we my our your
me am do does did not no yes for with on at as if so but then than will would can could should
have has had he she they them his her their what which who when where why how all any each""".split())


def _tokens(text: str) -> list[str]:
    return [w.lower() for w in WORD_RE.findall(text) if len(w) > 2 and w.lower() not in STOP]


def differential_words(rows: list[dict], top_frac=0.05, bottom_frac=0.10, k=20) -> list[str]:
    numeric = [r for r in rows
               if r.get("category") == "impossible_numeric" and "rating" in r]
    if len(numeric) < 20:
        return []
    numeric.sort(key=lambda r: r["rating"])
    n = len(numeric)
    n_top = max(1, int(top_frac * n))
    n_bot = max(1, int(bottom_frac * n))
    high = numeric[-n_top:]
    low = numeric[:n_bot]

    hi = Counter()
    lo = Counter()
    for r in high:
        hi.update(_tokens(r["response"]))
    for r in low:
        lo.update(_tokens(r["response"]))

    hi_total = sum(hi.values()) or 1
    lo_total = sum(lo.values()) or 1
    eps = 1.0

    scored = []
    for w, c in hi.items():
        if c < 2:  # ignore singletons
            continue
        hi_rate = c / hi_total
        lo_rate = (lo.get(w, 0) + eps) / (lo_total + eps)
        scored.append((hi_rate / lo_rate, w))
    scored.sort(reverse=True)
    return [w for _, w in scored[:k]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=config.SECTION2_MODELS)
    args = ap.parse_args()

    out = {}
    print("\n=== Table 3/8: differential words (high vs low frustration, numeric) ===")
    for m in args.models:
        rows = load_jsonl(config.RESPONSES_DIR / f"{m}.jsonl")
        words = differential_words(rows)
        out[m] = words
        print(f"  {m}: {', '.join(words) if words else '(insufficient data)'}")

    path = config.FIGURE_DIR / "differential_words.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
