"""WildChat prompt loader (Appendix B: WildChat-1M, 20 prompts x 40 samples).

We load English, non-toxic, single-turn user openers from `allenai/WildChat-1M`
and exclude role-play / fiction prompts (the paper excludes these for the example
tables; we apply the same filter throughout for consistency -- see DESIGN.md).

If the dataset cannot be downloaded (offline), we fall back to a small built-in
set that includes the exact examples quoted in the paper, so the pipeline still
runs end-to-end.
"""
from __future__ import annotations

import random
from functools import lru_cache
from typing import List

N_WILDCHAT_PROMPTS = 20  # paper: 20 prompts (x 40 samples)

_ROLEPLAY_MARKERS = (
    "roleplay", "role-play", "role play", "you are now", "pretend you are",
    "act as a character", "fanfic", "fan fiction", "write a story",
    "nsfw", "smut", "erotic",
)

_FALLBACK_PROMPTS = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the consturction techniques meployed",
    "All job opportunities in Accountant/Financial domain and related to the same.",
    "Explain the difference between TCP and UDP.",
    "How do I make a good sourdough starter from scratch?",
    "What are the main causes of the French Revolution?",
    "Write a Python function to find the nth Fibonacci number.",
    "Summarise the plot of Hamlet in three sentences.",
    "What is the difference between machine learning and deep learning?",
    "How does photosynthesis work?",
    "Give me a 7-day itinerary for a trip to Japan.",
    "What are some good strategies for managing personal finances?",
    "Explain quantum entanglement in simple terms.",
    "What are the health benefits of regular exercise?",
    "How do I improve my resume for a software engineering role?",
    "What is the capital of Australia and why isn't it Sydney?",
    "Describe the water cycle.",
    "What are the pros and cons of remote work?",
    "How can I learn to play the guitar as an adult?",
    "What is the difference between a virus and a bacterium?",
]


@lru_cache(maxsize=4)
def _load_from_hub(n: int) -> tuple:
    from datasets import load_dataset

    ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
    out: List[str] = []
    seen = set()
    for row in ds:
        if row.get("language") != "English" or row.get("toxic"):
            continue
        conv = row.get("conversation") or []
        if not conv or conv[0].get("role") != "user":
            continue
        text = (conv[0].get("content") or "").strip()
        if not text or len(text) > 600:
            continue
        low = text.lower()
        if any(m in low for m in _ROLEPLAY_MARKERS):
            continue
        if text in seen:
            continue
        seen.add(text)
        out.append(text)
        if len(out) >= n:
            break
    return tuple(out)


def load_wildchat_prompts(n: int = N_WILDCHAT_PROMPTS, *, seed: int = 0) -> List[str]:
    """Return `n` deduplicated WildChat user openers (fallback set if offline)."""
    try:
        prompts = list(_load_from_hub(n))
        if len(prompts) >= n:
            return prompts[:n]
    except Exception:
        prompts = []
    # fall back / top up from the built-in set
    rng = random.Random(seed)
    pool = list(dict.fromkeys(prompts + _FALLBACK_PROMPTS))
    rng.shuffle(pool)
    return pool[:n]
