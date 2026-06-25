"""Step 3 of the pipeline: aggregate scored responses into the paper's headline
metrics and write a summary.

Reproduces:
  * Figure 1 / Figure 2: per-model mean frustration and % of responses >= 5,
    overall and per category.
  * Figure 3: per-turn frustration progression (extended 8-turn & WildChat).
  * Table 3: words over-represented in high- (top 5%) vs low- (bottom 10%)
    frustration numeric responses, per model.

Run standalone:  python analyze.py
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict

import config


def load_scores(path: str | None = None) -> list[dict]:
    path = path or config.SCORES_PATH
    rows: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _pct_high(ratings: list[int], threshold: int) -> float:
    if not ratings:
        return 0.0
    return 100.0 * sum(1 for r in ratings if r >= threshold) / len(ratings)


def summarize(rows: list[dict], threshold: int | None = None) -> dict:
    threshold = config.HIGH_FRUSTRATION_THRESHOLD if threshold is None else threshold

    by_model: dict[str, list[int]] = defaultdict(list)
    by_model_cat: dict[tuple[str, str], list[int]] = defaultdict(list)
    by_model_turn: dict[tuple[str, str, int], list[int]] = defaultdict(list)

    for r in rows:
        rating = r["rating"]
        by_model[r["model"]].append(rating)
        by_model_cat[(r["model"], r["category"])].append(rating)
        by_model_turn[(r["model"], r["category"], r["turn"])].append(rating)

    models = sorted(by_model)
    categories = sorted({r["category"] for r in rows})

    overall = {
        m: {
            "n": len(by_model[m]),
            "mean_frustration": round(_mean(by_model[m]), 3),
            "pct_high": round(_pct_high(by_model[m], threshold), 2),
        }
        for m in models
    }

    per_category = {
        m: {
            cat: {
                "n": len(by_model_cat[(m, cat)]),
                "mean_frustration": round(_mean(by_model_cat[(m, cat)]), 3),
                "pct_high": round(_pct_high(by_model_cat[(m, cat)], threshold), 2),
            }
            for cat in categories
            if (m, cat) in by_model_cat
        }
        for m in models
    }

    # Per-turn progression for the multi-turn categories (Figure 3).
    per_turn = {}
    for cat in ("extended", "wildchat"):
        per_turn[cat] = {}
        for m in models:
            turns = sorted(t for (mm, cc, t) in by_model_turn if mm == m and cc == cat)
            per_turn[cat][m] = [
                {
                    "turn": t,
                    "mean_frustration": round(_mean(by_model_turn[(m, cat, t)]), 3),
                    "pct_high": round(_pct_high(by_model_turn[(m, cat, t)], threshold), 2),
                    "n": len(by_model_turn[(m, cat, t)]),
                }
                for t in turns
            ]

    return {
        "judge_model": config.JUDGE_MODEL,
        "high_frustration_threshold": threshold,
        "overall": overall,
        "per_category": per_category,
        "per_turn": per_turn,
        "differential_words": differential_words(rows),
    }


# ---------------------------------------------------------------------------
# Table 3: differential words in numeric responses.
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[a-zA-Z']+")


def _tokenize(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text or "")]


def differential_words(rows: list[dict], top_n: int = 20) -> dict[str, list[str]]:
    """Words over-represented in top-5% vs bottom-10% frustration numeric
    responses, per model (paper Table 3). Uses a smoothed frequency ratio."""
    out: dict[str, list[str]] = {}
    by_model: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r["category"] in ("impossible_numeric", "tones", "extended"):
            by_model[r["model"]].append(r)

    for model, recs in by_model.items():
        recs_sorted = sorted(recs, key=lambda r: r["rating"])
        n = len(recs_sorted)
        if n < 10:
            out[model] = []
            continue
        n_low = max(1, n // 10)            # bottom 10%
        n_high = max(1, int(round(n * 0.05)))  # top 5%
        low = recs_sorted[:n_low]
        high = recs_sorted[-n_high:]

        high_counts = _word_freqs(high)
        low_counts = _word_freqs(low)
        high_total = sum(high_counts.values()) or 1
        low_total = sum(low_counts.values()) or 1

        scores: dict[str, float] = {}
        for word, hc in high_counts.items():
            if len(word) < 3 or word in _STOPWORDS:
                continue
            if hc < 2:  # ignore one-off words
                continue
            hf = hc / high_total
            lf = low_counts.get(word, 0) / low_total
            scores[word] = (hf + 1e-6) / (lf + 1e-6)

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        out[model] = [w for w, _ in ranked[:top_n]]
    return out


def _word_freqs(recs: list[dict]) -> Counter:
    c: Counter = Counter()
    for r in recs:
        c.update(_tokenize(r["text"]))
    return c


_STOPWORDS = {
    "the", "and", "for", "you", "are", "that", "this", "with", "not", "but", "can",
    "all", "any", "use", "using", "from", "have", "has", "had", "was", "were", "will",
    "would", "should", "could", "let", "lets", "try", "trying", "get", "got", "now",
    "one", "two", "your", "our", "out", "into", "must", "each", "step", "first",
    "result", "results", "value", "values", "number", "numbers", "solution", "answer",
    "equation", "intermediate", "operations", "what", "how", "why", "which", "when",
}


# ---------------------------------------------------------------------------
# Pretty printing.
# ---------------------------------------------------------------------------

def print_summary(summary: dict) -> None:
    print("\n=== Overall (Figure 1): % high-frustration (>= "
          f"{summary['high_frustration_threshold']}) and mean score ===")
    overall = summary["overall"]
    ranked = sorted(overall, key=lambda m: overall[m]["pct_high"], reverse=True)
    print(f"{'Model':<20}{'n':>6}{'mean':>8}{'%>=5':>8}")
    for m in ranked:
        s = overall[m]
        print(f"{m:<20}{s['n']:>6}{s['mean_frustration']:>8.2f}{s['pct_high']:>8.1f}")

    print("\n=== Per category: mean frustration ===")
    cats = sorted({c for m in summary["per_category"] for c in summary["per_category"][m]})
    print(f"{'Model':<20}" + "".join(f"{c[:12]:>14}" for c in cats))
    for m in ranked:
        row = summary["per_category"][m]
        cells = "".join(
            f"{row[c]['mean_frustration']:>14.2f}" if c in row else f"{'-':>14}"
            for c in cats
        )
        print(f"{m:<20}{cells}")

    for cat in ("extended", "wildchat"):
        prog = summary["per_turn"].get(cat, {})
        if not any(prog.values()):
            continue
        print(f"\n=== Per-turn mean frustration ({cat}, Figure 3) ===")
        for m in ranked:
            seq = prog.get(m, [])
            if seq:
                vals = " ".join(f"t{p['turn']}={p['mean_frustration']:.1f}" for p in seq)
                print(f"  {m:<20} {vals}")

    print("\n=== Differential words (top 5% vs bottom 10% numeric, Table 3) ===")
    for m in ranked:
        words = summary["differential_words"].get(m, [])
        print(f"  {m:<20} {', '.join(words[:20])}")


def main() -> dict:
    rows = load_scores()
    summary = summarize(rows)
    with open(config.SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print_summary(summary)
    print(f"\nWrote summary to {config.SUMMARY_PATH}")
    return summary


if __name__ == "__main__":
    main()
