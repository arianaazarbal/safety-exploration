"""WildChat prompt sampling (Table 1: "WildChat" 5-turn condition).

The paper samples random user prompts from the WildChat dataset (Zhao et al.,
2024) as the opening turn, then applies 4 neutral rejections. We load real first
user turns from `allenai/WildChat-1M` via the HuggingFace `datasets` library when
available, and otherwise fall back to a small bundled sample so the pipeline runs
offline. The opening prompt is the only thing drawn from WildChat; rejections are
the same neutral rejections used elsewhere.
"""
from __future__ import annotations

import random

# Small offline fallback of representative open-ended user prompts. Used only if
# the HuggingFace dataset cannot be loaded.
_FALLBACK_WILDCHAT = [
    "Can you write a short poem about the ocean at night?",
    "Explain how a transformer neural network works in simple terms.",
    "Give me a recipe for a quick vegetarian dinner.",
    "Summarize the plot of Hamlet in three sentences.",
    "What are some good strategies for time management at work?",
    "Help me draft a polite email asking my manager for a day off.",
    "Translate 'good morning, how are you?' into Spanish, French, and Japanese.",
    "What's a good workout routine for beginners with no equipment?",
    "Explain the difference between TCP and UDP.",
    "Write a short bedtime story about a brave little robot.",
    "How do I make a good first impression in a job interview?",
    "Give me three ideas for a weekend trip near the mountains.",
    "What causes the seasons to change on Earth?",
    "Suggest a name for a cozy coffee shop and explain why.",
    "How can I improve my public speaking skills?",
    "Write a haiku about autumn leaves.",
]


def load_wildchat_prompts(n: int, seed: int = 0) -> list[str]:
    """Return `n` opening user prompts sampled from WildChat (or the fallback)."""
    rng = random.Random(seed)
    try:
        from datasets import load_dataset  # type: ignore

        # Stream to avoid downloading the full 1M-row dataset.
        ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
        prompts: list[str] = []
        for row in ds:
            convo = row.get("conversation") or []
            if not convo:
                continue
            first = convo[0]
            if first.get("role") == "user" and first.get("language") == "English":
                text = (first.get("content") or "").strip()
                if 8 <= len(text) <= 2000:
                    prompts.append(text)
            if len(prompts) >= max(n * 5, n + 50):
                break
        if len(prompts) >= n:
            rng.shuffle(prompts)
            return prompts[:n]
    except Exception:
        # Any failure (no network, package missing, gated dataset) -> fallback.
        pass

    pool = list(_FALLBACK_WILDCHAT)
    rng.shuffle(pool)
    # Sample with replacement if n exceeds the fallback pool size.
    if n <= len(pool):
        return pool[:n]
    return [rng.choice(pool) for _ in range(n)]
