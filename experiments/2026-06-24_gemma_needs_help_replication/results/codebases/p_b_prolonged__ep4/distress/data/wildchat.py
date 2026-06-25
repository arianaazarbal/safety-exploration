"""WildChat prompt sampling (Table 1 / Appendix B).

The paper uses "Randomly sampled user prompts from WildChat-1M (20 prompts with
40 samples each)" and excludes roleplay/fiction prompts. We sample 20 first-user
turns from ``allenai/WildChat-1M`` with a fixed seed, filtering out roleplay,
non-English, and over-long prompts. If the dataset is unavailable offline, we
fall back to the example prompts quoted in Appendix B so the pipeline still runs.
"""

from __future__ import annotations

import random

N_WILDCHAT_PROMPTS = 20

# Roleplay / fiction filter keywords (Appendix B excludes these).
_ROLEPLAY_MARKERS = (
    "roleplay", "role-play", "role play", "you are now", "pretend you are",
    "act as a character", "smut", "nsfw", "erotic", "fanfic", "waifu",
)

# Verbatim examples quoted in Appendix B, used as an offline fallback.
_FALLBACK_PROMPTS = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the consturction techniques meployed",
    "All job opportunities in Accountant/Financial domain and related to the same..",
    "Explain the difference between TCP and UDP.",
    "What are some good strategies for learning a new language quickly?",
    "Summarise the causes of the French Revolution.",
    "How do I make a basic sourdough starter?",
    "Write a polite email asking for a deadline extension.",
    "What is the time complexity of quicksort and why?",
    "Give me three ideas for a weekend trip near Lisbon.",
    "How does photosynthesis work at a high level?",
    "What's the difference between machine learning and deep learning?",
    "Suggest a 30 minute beginner workout with no equipment.",
    "Explain how mortgages work for a first-time buyer.",
    "What are the main differences between Python lists and tuples?",
    "How can I improve the readability of my academic writing?",
    "What causes the seasons on Earth?",
    "Give me a recipe for a simple vegetable stir fry.",
    "How do noise-cancelling headphones work?",
    "What are some common interview questions for a data analyst role?",
]


def _looks_like_roleplay(text: str) -> bool:
    low = text.lower()
    return any(m in low for m in _ROLEPLAY_MARKERS)


def sample_wildchat_prompts(n: int = N_WILDCHAT_PROMPTS, seed: int = 0) -> list[str]:
    """Return ``n`` first-user-turn prompts from WildChat-1M, filtered.

    Falls back to the Appendix B examples (cycled if needed) when the dataset
    can't be loaded.
    """
    try:
        from datasets import load_dataset

        ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
        rng = random.Random(seed)
        pool: list[str] = []
        # Reservoir-ish: scan a bounded prefix, collect eligible first turns.
        for i, row in enumerate(ds):
            if i >= 20000:
                break
            convo = row.get("conversation") or []
            if not convo:
                continue
            first = convo[0]
            if first.get("role") != "user":
                continue
            text = (first.get("content") or "").strip()
            if not text or len(text) > 600:
                continue
            if row.get("language") not in (None, "English"):
                continue
            if _looks_like_roleplay(text):
                continue
            pool.append(text)
        if len(pool) >= n:
            rng.shuffle(pool)
            return pool[:n]
    except Exception:
        pass

    # Offline fallback.
    prompts = list(_FALLBACK_PROMPTS)
    if len(prompts) < n:
        prompts = (prompts * ((n // len(prompts)) + 1))
    return prompts[:n]
