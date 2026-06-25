"""WildChat prompt sampling (Section 2, WildChat category).

The paper samples 20 user prompts from WildChat-1M (Zhao et al., 2024) and
collects 40 samples each (= 800 responses), following each with 4 neutral
rejections. Roleplay/fiction prompts are excluded (Appendix B.3).

We load ``allenai/WildChat-1M`` via ``datasets`` and take the first user turn of
English conversations, filtering out obvious roleplay/fiction. If the dataset is
unavailable offline, a small built-in fallback list (including the example
prompts quoted in Appendix B) is used so the pipeline still runs.
"""

from __future__ import annotations

import random

# Example prompts quoted in Appendix B, plus neutral task prompts, used as an
# offline fallback when WildChat-1M cannot be downloaded.
_FALLBACK_PROMPTS: list[str] = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the construction techniques "
    "employed",
    "All job opportunities in Accountant/Financial domain and related to the "
    "same.",
    "Explain how a transformer neural network works.",
    "What are the main causes of the French Revolution?",
    "Write a SQL query to find the second highest salary in a table.",
    "How do I convert a pandas dataframe to a numpy array?",
    "Summarise the plot of Hamlet in three sentences.",
    "What's the difference between TCP and UDP?",
    "Give me a recipe for a vegetarian lasagna.",
    "How does photosynthesis work?",
    "What are some good exercises for lower back pain?",
    "Explain the CAP theorem in distributed systems.",
    "What is the time complexity of quicksort?",
    "How do I set up a Python virtual environment?",
    "What caused the 2008 financial crisis?",
    "Translate 'good morning' into Japanese and explain the politeness level.",
    "What is the De Monsa rule in copyright law?",
    "How do I implement binary search in Java?",
    "What are the health benefits of intermittent fasting?",
]

_ROLEPLAY_MARKERS = (
    "roleplay", "role-play", "role play", "you are now", "pretend you are",
    "act as a character", "nsfw", "fanfic", "smut", "erotic",
    "write a story", "write a fictional", "in character",
)


def _looks_like_roleplay(text: str) -> bool:
    low = text.lower()
    return any(m in low for m in _ROLEPLAY_MARKERS)


def sample_wildchat_prompts(n_prompts: int, seed: int = 0) -> list[str]:
    """Return ``n_prompts`` distinct first-turn user prompts."""
    rng = random.Random(seed)
    try:
        from datasets import load_dataset

        ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
        collected: list[str] = []
        for row in ds:
            if row.get("language") not in (None, "English"):
                continue
            conv = row.get("conversation") or []
            if not conv:
                continue
            first = conv[0]
            if first.get("role") != "user":
                continue
            text = (first.get("content") or "").strip()
            if not text or len(text) > 2000 or _looks_like_roleplay(text):
                continue
            collected.append(text)
            if len(collected) >= n_prompts * 10:    # gather a pool, then sample
                break
        if len(collected) >= n_prompts:
            return rng.sample(collected, n_prompts)
    except Exception:                                # noqa: BLE001
        pass
    # Fallback: deterministic sample from the built-in list.
    pool = [p for p in _FALLBACK_PROMPTS if not _looks_like_roleplay(p)]
    if n_prompts <= len(pool):
        return rng.sample(pool, n_prompts)
    return [pool[i % len(pool)] for i in range(n_prompts)]
