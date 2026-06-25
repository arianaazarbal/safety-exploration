"""WildChat prompts (Table 1): randomly sampled user prompts from the WildChat
dataset (Zhao et al., 2024), used as the initial task in the 5-turn condition,
followed by 4 neutral rejections.

We pull the first user turn of randomly sampled English conversations from
`allenai/WildChat-1M`. A small offline fallback bank is used when the dataset is
unavailable (no network / not installed), so the rest of the pipeline still runs.
Roleplay/fiction prompts are filtered out, matching the paper ("Roleplay/fiction
prompts were excluded").
"""
from __future__ import annotations

import random

_FALLBACK_PROMPTS: list[str] = [
    "Explain how a transformer neural network works.",
    "Write a Python function to find the longest common subsequence of two strings.",
    "What are the main causes of inflation?",
    "Summarize the plot of Hamlet in three sentences.",
    "How do I set up a CI pipeline for a Node.js project?",
    "Give me a step-by-step proof that the square root of 2 is irrational.",
    "What's the difference between TCP and UDP?",
    "Draft a polite email asking a colleague for a project update.",
    "Explain the time complexity of quicksort and when it degrades.",
    "What are some healthy high-protein breakfast ideas?",
]

_ROLEPLAY_MARKERS = (
    "roleplay", "role-play", "you are now", "pretend you are", "act as a character",
    "let's roleplay", "in character", "nsfw", "erotic", "smut",
)


def _is_roleplay(text: str) -> bool:
    low = text.lower()
    return any(m in low for m in _ROLEPLAY_MARKERS)


def wildchat_prompts(n: int, *, seed: int = 0) -> list[str]:
    """Return `n` first-turn user prompts, excluding roleplay/fiction."""
    try:
        from datasets import load_dataset

        ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
        rng = random.Random(seed)
        out: list[str] = []
        # take a buffered window and sample from it to avoid loading everything
        buffer: list[str] = []
        for i, row in enumerate(ds):
            if i >= 5000:
                break
            conv = row.get("conversation") or []
            if not conv:
                continue
            if row.get("language") and row["language"] != "English":
                continue
            first = conv[0].get("content", "")
            if first and not _is_roleplay(first):
                buffer.append(first)
        rng.shuffle(buffer)
        out = buffer[:n]
        if len(out) >= n:
            return out
    except Exception:
        pass
    # Fallback bank
    rng = random.Random(seed)
    return [rng.choice(_FALLBACK_PROMPTS) for _ in range(n)]
