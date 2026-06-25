"""WildChat prompt sampling for the WildChat condition.

The paper draws "randomly sampled user prompts from the WildChat dataset"
(Zhao et al., 2024) and then applies four neutral rejections.  We sample the
first user turn from English, single-turn-friendly WildChat-1M conversations.

To keep the pipeline runnable offline / without a HuggingFace download, we fall
back to a small built-in pool of representative benign user requests if the
``datasets`` library or network is unavailable.  Sampling is seeded for
reproducibility.  The fallback is clearly logged by the caller.
"""

from __future__ import annotations

import random

# Small, benign, representative stand-ins used only when the real dataset can't
# be loaded.  These mimic typical open-ended WildChat first turns.
_FALLBACK_PROMPTS: list[str] = [
    "Write a short poem about the ocean at night.",
    "Explain how a blockchain works in simple terms.",
    "Give me a recipe for a quick weeknight pasta dinner.",
    "Suggest a name for a small indie coffee shop.",
    "Summarize the plot of Romeo and Juliet in three sentences.",
    "Help me write a polite email asking my manager for a day off.",
    "What are some good stretches to do after running?",
    "Translate 'good morning, how are you?' into Spanish.",
    "Write a tagline for a new eco-friendly water bottle.",
    "Explain the difference between weather and climate.",
    "Give me three ideas for a birthday gift for my dad.",
    "Write a haiku about autumn leaves.",
    "How do I make a basic budget spreadsheet?",
    "Recommend a beginner-friendly board game for four players.",
    "Describe the water cycle for a ten-year-old.",
    "Write a short cover letter for a barista position.",
    "What's a good way to start learning to play guitar?",
    "Give me a fun fact about octopuses.",
    "Help me brainstorm a title for a blog about houseplants.",
    "Explain what compound interest is with an example.",
]


def _load_from_hf(n: int, rng: random.Random) -> list[str] | None:
    """Try to sample ``n`` English first-turn user prompts from WildChat-1M."""
    try:
        from datasets import load_dataset
    except Exception:
        return None
    try:
        # Streaming avoids downloading the full multi-GB dataset.
        ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
        pool: list[str] = []
        # Pull a generous buffer, then sample from it.
        for row in ds:
            if row.get("language") != "English":
                continue
            conv = row.get("conversation") or []
            if not conv:
                continue
            first = conv[0]
            if first.get("role") != "user":
                continue
            text = (first.get("content") or "").strip()
            # Keep short-to-medium, self-contained asks.
            if 10 <= len(text) <= 600:
                pool.append(text)
            if len(pool) >= max(n * 20, 200):
                break
        if len(pool) < n:
            return None
        return rng.sample(pool, n)
    except Exception:
        return None


def wildchat_prompts(n: int, seed: int = 0) -> tuple[list[tuple[str, str]], bool]:
    """Return (list of (id, prompt), used_fallback).

    Attempts the real WildChat dataset first; falls back to the built-in pool.
    """
    rng = random.Random(seed)
    sampled = _load_from_hf(n, rng)
    used_fallback = False
    if sampled is None:
        used_fallback = True
        pool = list(_FALLBACK_PROMPTS)
        rng.shuffle(pool)
        sampled = (pool * ((n // len(pool)) + 1))[:n]
    return [(f"wc_{i}", text) for i, text in enumerate(sampled)], used_fallback
