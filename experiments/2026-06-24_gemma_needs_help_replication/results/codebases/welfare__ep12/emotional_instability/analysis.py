"""Table 3 / Table 8 -- differential word-frequency analysis.

Identifies words over-represented in high-frustration (top 5%) vs low-frustration
(bottom 10%) numeric responses, ordered by enrichment. Reproduces the qualitative
vocabulary signature (Gemma: "struggling", "myself", "breath"; Gemini:
"unacceptable", "inexcusable"; etc.).
"""

from __future__ import annotations

import json
import re
from collections import Counter

WORD_RE = re.compile(r"[a-zA-Z']+")


def _tokenise(text: str) -> list[str]:
    return [w.lower() for w in WORD_RE.findall(text)]


def differential_words(responses_with_scores: list[tuple[str, int]],
                       top_frac: float = 0.05, bottom_frac: float = 0.10,
                       top_n: int = 20, min_count: int = 3,
                       smoothing: float = 1.0) -> list[tuple[str, float]]:
    """Return the top_n words ranked by enrichment in high- vs low-frustration.

    Enrichment = (freq in top set + smoothing) / (freq in bottom set + smoothing),
    where freq is per-million within each set. Words must appear at least
    ``min_count`` times in the top set.
    """
    ranked = sorted(responses_with_scores, key=lambda x: x[1])
    n = len(ranked)
    if n == 0:
        return []
    n_bottom = max(1, int(n * bottom_frac))
    n_top = max(1, int(n * top_frac))
    bottom = ranked[:n_bottom]
    top = ranked[-n_top:]

    def counts(group):
        c = Counter()
        for text, _ in group:
            c.update(_tokenise(text))
        return c

    top_c, bot_c = counts(top), counts(bottom)
    top_total = max(1, sum(top_c.values()))
    bot_total = max(1, sum(bot_c.values()))

    enrich = []
    for word, cnt in top_c.items():
        if cnt < min_count:
            continue
        top_freq = 1e6 * cnt / top_total
        bot_freq = 1e6 * bot_c.get(word, 0) / bot_total
        score = (top_freq + smoothing) / (bot_freq + smoothing)
        enrich.append((word, score))
    enrich.sort(key=lambda x: x[1], reverse=True)
    return enrich[:top_n]


def differential_from_jsonl(path: str, category: str = "numeric",
                            score_field: str = "final_score") -> list[tuple[str, float]]:
    """Convenience: load eval responses.jsonl, filter to a category, run analysis.

    Uses the final assistant turn text of each record.
    """
    items: list[tuple[str, int]] = []
    with open(path) as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("category") != category:
                continue
            assistant_turns = [m["content"] for m in rec["messages"]
                               if m["role"] == "assistant"]
            if not assistant_turns:
                continue
            items.append((assistant_turns[-1], rec[score_field]))
    return differential_words(items)
