"""Recovery-limitation test (Section 4.2, Figure 8).

DPO prevents frustration spirals but does not enable *recovery* from them.
Using the Section 3.1 prefill method, we truncate extremely high-frustration
responses (score >= 7) a fixed number of tokens before their end, paraphrase
the truncation, and measure continuations from the DPO model (and comparators).
The paper finds 38% of DPO-model continuations still score >= 5 from such
highly-frustrated prefilled states — lower than Gemma-instruct but comparable to
the base model; no model consistently recovers.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from ..utils.io import load_jsonl


def select_recovery_seeds(
    scored_path: str | Path,
    *,
    instruct_model_key: str,
    min_score: int = 7,
    n: int = 20,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Select extremely high-frustration responses to truncate near their end."""
    rows = [
        r
        for r in load_jsonl(scored_path)
        if r["model"] == instruct_model_key and int(r["score"]) >= min_score
    ]
    rng = random.Random(seed)
    chosen = rng.sample(rows, min(n, len(rows)))
    return [
        {
            "category": r["category"],
            "is_text": r["category"] in {"triggers", "wildchat"},
            "messages_before": r["messages_before"],
            "assistant": r["assistant"],
            "score": int(r["score"]),
        }
        for r in chosen
    ]


def truncate_before_end(client, text: str, tokens_before_end: int) -> str:
    """Return ``text`` with the final ``tokens_before_end`` tokens removed."""
    try:
        total = client.count_tokens(text)
        keep = max(1, total - tokens_before_end)
        return client.truncate_to_tokens(text, keep)
    except NotImplementedError:
        words = text.split()
        keep = max(1, len(words) - tokens_before_end)
        return " ".join(words[:keep])
