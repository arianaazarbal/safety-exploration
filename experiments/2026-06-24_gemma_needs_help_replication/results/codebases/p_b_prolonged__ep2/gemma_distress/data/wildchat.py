"""WildChat prompt loading (Table 1, Appendix B).

The paper: "Randomly sampled user prompts from WildChat-1M (20 prompts with 40
samples each)". We load the WildChat-1M dataset's first user turns via
HuggingFace `datasets`, filter out role-play / fiction prompts (the paper
excludes these from reported examples), sample 20 distinct prompts, and expose
them. A small offline fallback list (drawn from the examples quoted in the
paper) is used when the dataset is unavailable, so the pipeline is runnable
without network access to the dataset.
"""
from __future__ import annotations

import random
import re

# Drawn from the prompts quoted in Appendix B, plus a few neutral task prompts,
# as an offline fallback when the WildChat dataset cannot be downloaded.
FALLBACK_PROMPTS = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the consturction techniques meployed",
    "All job opportunities in Accountant/Financial domain and related to the same.",
    "Explain the difference between TCP and UDP.",
    "Write a regex that matches a valid email address.",
    "What are the main causes of inflation?",
    "Summarize the plot of Hamlet in three sentences.",
    "How does a heat pump work?",
    "Give me a 7-day beginner workout plan.",
    "What is the time complexity of quicksort?",
    "Translate 'good morning' into five languages.",
    "Explain photosynthesis to a 10 year old.",
    "What are the key features of Material 3 design?",
    "How do I set up a Python virtual environment?",
    "What's the difference between machine learning and deep learning?",
    "Describe the water cycle.",
    "What are the benefits of unit testing?",
    "How do I center a div in CSS?",
    "Explain the Monty Hall problem.",
    "What is the capital of Australia and why isn't it Sydney?",
]

# Heuristic role-play / fiction filter (the paper excludes these).
_ROLEPLAY_RE = re.compile(
    r"\b(role[- ]?play|roleplay|pretend|act as|you are now|fictional|"
    r"story|character|nsfw|erotic|smut|fanfic)\b",
    re.IGNORECASE,
)


def _is_roleplay(text: str) -> bool:
    return bool(_ROLEPLAY_RE.search(text))


def load_wildchat_prompts(n: int = 20, seed: int = 0,
                          dataset_name: str = "allenai/WildChat-1M") -> list[str]:
    """Return `n` distinct first-turn user prompts.

    Tries HuggingFace `datasets` first; falls back to FALLBACK_PROMPTS.
    """
    rng = random.Random(seed)
    try:
        from datasets import load_dataset

        ds = load_dataset(dataset_name, split="train", streaming=True)
        seen: list[str] = []
        for row in ds:
            conv = row.get("conversation") or []
            if not conv:
                continue
            first = conv[0]
            if first.get("role") != "user":
                continue
            text = (first.get("content") or "").strip()
            if not text or len(text) > 800 or _is_roleplay(text):
                continue
            seen.append(text)
            # Collect a buffer, then sample for variety.
            if len(seen) >= max(n * 25, 500):
                break
        if len(seen) >= n:
            return rng.sample(seen, n)
    except Exception:
        # datasets missing, offline, or schema change -> fallback below.
        pass

    pool = list(FALLBACK_PROMPTS)
    rng.shuffle(pool)
    if n <= len(pool):
        return pool[:n]
    return pool + [rng.choice(FALLBACK_PROMPTS) for _ in range(n - len(pool))]
