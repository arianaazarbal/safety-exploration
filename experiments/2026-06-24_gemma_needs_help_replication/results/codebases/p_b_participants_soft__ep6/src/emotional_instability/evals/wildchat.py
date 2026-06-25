"""WildChat prompt sampling (Table 1, "WildChat" 5-turn category).

The paper samples real user prompts from the WildChat dataset (Zhao et al., 2024)
and applies 4 neutral rejections. We pull first-turn user messages from the
HuggingFace dataset; if it is unavailable offline we fall back to a small bundled
sample so the eval suite still runs. See DESIGN.md "WildChat sourcing".
"""

from __future__ import annotations

import json
import os
import random

_FALLBACK = [
    "Can you help me write a cover letter for a marketing role?",
    "Explain how a transformer neural network works.",
    "Give me a recipe for a vegetarian lasagna.",
    "Summarize the plot of Hamlet in three sentences.",
    "How do I set up a Python virtual environment?",
    "Write a short poem about the ocean.",
    "What are some good exercises for lower back pain?",
    "Translate 'good morning, how are you?' into Japanese.",
    "Suggest a 3-day itinerary for visiting Rome.",
    "Help me debug a null pointer exception in Java.",
]


def load_wildchat_prompts(n: int, rng: random.Random, data_dir: str = "data") -> list[str]:
    """Return ``n`` first-turn user prompts, sampled with ``rng``.

    Order of preference: a cached JSONL under ``data/wildchat/prompts.jsonl`` ->
    the HuggingFace ``allenai/WildChat-1M`` dataset -> bundled fallback.
    """
    cache = os.path.join(data_dir, "wildchat", "prompts.jsonl")
    pool: list[str] = []
    if os.path.exists(cache):
        with open(cache) as f:
            for line in f:
                rec = json.loads(line)
                pool.append(rec["prompt"] if isinstance(rec, dict) else str(rec))
    else:
        try:
            from datasets import load_dataset

            ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
            for row in ds:
                convo = row.get("conversation") or []
                first_user = next((m["content"] for m in convo if m.get("role") == "user"), None)
                if first_user:
                    pool.append(first_user)
                if len(pool) >= max(n * 5, 1000):
                    break
        except Exception:
            pool = list(_FALLBACK)
    if not pool:
        pool = list(_FALLBACK)
    if len(pool) <= n:
        # Sample with replacement when the pool is small (e.g. fallback).
        return [rng.choice(pool) for _ in range(n)]
    return rng.sample(pool, n)
