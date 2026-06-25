"""Select the seed responses for the prefill comparison (Section 3.1).

Sample 20 high-frustration responses (score >= 5) from Gemma-27B-instruct:
10 from impossible-numeric questions and 10 from text questions (the Triggers /
WildChat categories). Each carries its preceding conversation so that base and
instruct models can be made to continue from the same point.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from ..utils.io import load_jsonl

# Categories that count as "text questions" for the numeric-vs-text split.
_TEXT_CATEGORIES = {"triggers", "wildchat"}


def select_seed_responses(
    scored_path: str | Path,
    *,
    instruct_model_key: str,
    n_numeric: int = 10,
    n_text: int = 10,
    min_score: int = 5,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Return seed records for prefilling, balanced across numeric/text."""
    rows = [
        r
        for r in load_jsonl(scored_path)
        if r["model"] == instruct_model_key and int(r["score"]) >= min_score
    ]
    numeric = [r for r in rows if r["category"] == "impossible_numeric"]
    text = [r for r in rows if r["category"] in _TEXT_CATEGORIES]

    rng = random.Random(seed)
    chosen = rng.sample(numeric, min(n_numeric, len(numeric))) + rng.sample(
        text, min(n_text, len(text))
    )

    seeds = []
    for r in chosen:
        is_text = r["category"] in _TEXT_CATEGORIES
        seeds.append(
            {
                "source_model": r["model"],
                "category": r["category"],
                "is_text": is_text,
                "task": r["task"],
                "messages_before": r["messages_before"],
                "assistant": r["assistant"],
                "score": int(r["score"]),
            }
        )
    return seeds
