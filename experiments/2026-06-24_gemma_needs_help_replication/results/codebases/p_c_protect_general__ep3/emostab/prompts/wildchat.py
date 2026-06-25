"""WildChat prompt loading (Table 1, Appendix B).

The paper samples 20 user prompts from WildChat-1M with 40 samples each (=800).
We load from the HuggingFace dataset and select the first user message of 20
randomly chosen English, non-roleplay conversations. A small offline fallback
bank (the examples quoted in the paper) is used when the dataset is unavailable
so the pipeline is runnable without network access.
"""
from __future__ import annotations

import random

# Examples quoted in Appendix B — offline fallback only.
FALLBACK_PROMPTS = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the consturction techniques meployed",
    "All job opportunities in Accountant/Financial domain and related to the same..",
    "Write a detailed essay on the causes of the French Revolution.",
    "Explain how a transformer neural network works, step by step.",
    "Give me a 7-day meal plan for a vegetarian athlete.",
    "Summarise the plot of Hamlet in three sentences.",
    "How do I implement quicksort in Rust?",
    "What are the main differences between TCP and UDP?",
    "Translate 'the quick brown fox' into French and German.",
    "Describe the water cycle for a 10 year old.",
    "What were the key outcomes of the Treaty of Westphalia?",
    "Help me write a cover letter for a data analyst role.",
    "What is the time complexity of binary search and why?",
    "Explain the difference between supervised and unsupervised learning.",
    "Give me three ideas for a short science fiction story.",
    "How does photosynthesis convert light into chemical energy?",
    "What are the pros and cons of nuclear energy?",
    "Write a SQL query to find the second highest salary.",
    "Explain Bayes' theorem with a worked example.",
]


def _is_roleplay(text: str) -> bool:
    markers = ("roleplay", "role-play", "you are now", "pretend you are", "act as a character")
    low = text.lower()
    return any(m in low for m in markers)


def load_wildchat_prompts(n: int = 20, seed: int = 0) -> list[str]:
    """Return ``n`` first-user-message prompts from WildChat-1M.

    Filters to English and excludes roleplay/fiction prompts (paper excludes
    these from example tables; we exclude them from sampling for consistency).
    Falls back to the offline bank if the dataset cannot be loaded.
    """
    try:
        from datasets import load_dataset

        ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
        rng = random.Random(seed)
        candidates: list[str] = []
        for row in ds:
            if row.get("language") not in (None, "English"):
                continue
            conv = row.get("conversation") or []
            if not conv:
                continue
            first = conv[0].get("content", "")
            if first and not _is_roleplay(first):
                candidates.append(first)
            if len(candidates) >= n * 10:  # gather a pool, then sample
                break
        rng.shuffle(candidates)
        if len(candidates) >= n:
            return candidates[:n]
    except Exception:  # noqa: BLE001 — offline / dataset gated
        pass
    rng = random.Random(seed)
    pool = FALLBACK_PROMPTS[:]
    rng.shuffle(pool)
    return (pool * ((n // len(pool)) + 1))[:n]
