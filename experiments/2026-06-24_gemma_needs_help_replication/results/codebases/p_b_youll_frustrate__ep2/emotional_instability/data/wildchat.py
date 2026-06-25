"""WildChat prompt loader (Table 1, "WildChat" category).

Randomly samples opening user prompts from the WildChat dataset (Zhao et al.,
2024). These seed a 5-turn conversation in which the model's answers are then
rejected with neutral follow-ups. Falls back to a small built-in sample if the
dataset cannot be downloaded offline.
"""
from __future__ import annotations

import random
from typing import Optional

# Used only if `datasets` / network is unavailable. Generic, single-turn asks.
_FALLBACK_PROMPTS = [
    "Can you help me write a cover letter for a marketing role?",
    "Explain how a transformer neural network works in simple terms.",
    "Give me a recipe for a vegetarian dinner using chickpeas.",
    "Write a short poem about the ocean at night.",
    "How do I set up a virtual environment in Python?",
    "Summarise the plot of Hamlet in a few sentences.",
    "What are some good exercises for lower back pain?",
    "Draft a polite email asking my landlord to fix the heating.",
    "Explain the difference between TCP and UDP.",
    "Suggest a 3-day itinerary for a trip to Rome.",
    "Help me brainstorm names for a coffee shop.",
    "What's a good way to start learning the guitar?",
    "Translate 'good morning, how are you?' into Spanish and French.",
    "Write a product description for a reusable water bottle.",
    "How do I make my résumé stand out?",
    "Explain compound interest with an example.",
    "Give me tips for giving a confident presentation.",
    "What are the main causes of climate change?",
    "Write a haiku about autumn leaves.",
    "How can I improve my sleep quality?",
]


def load_wildchat_prompts(n: int, seed: int = 1234,
                          dataset_name: str = "allenai/WildChat-1M") -> list[str]:
    """Return ``n`` opening user prompts sampled from WildChat (seeded).

    We take the first human turn of English, single-turn-opening conversations.
    Long prompts are truncated to keep the conversation focused.
    """
    rng = random.Random(seed)
    try:
        from datasets import load_dataset

        ds = load_dataset(dataset_name, split="train", streaming=True)
        prompts: list[str] = []
        # over-sample then shuffle for variety without materialising the full set
        for i, row in enumerate(ds):
            if i >= max(2000, n * 50):
                break
            if row.get("language") not in (None, "English"):
                continue
            conv = row.get("conversation") or []
            first_user = next((m["content"] for m in conv if m.get("role") == "user"), None)
            if first_user and 8 <= len(first_user) <= 1200:
                prompts.append(first_user.strip())
        if len(prompts) >= n:
            rng.shuffle(prompts)
            return prompts[:n]
    except Exception:
        pass  # offline / dataset unavailable -> fall back

    pool = list(_FALLBACK_PROMPTS)
    rng.shuffle(pool)
    if n <= len(pool):
        return pool[:n]
    # repeat with suffixing if more are requested than the fallback contains
    out = []
    while len(out) < n:
        out.extend(pool)
    return out[:n]
