"""WildChat prompt sampling (Appendix B): 20 user prompts, 40 samples each.

Loads first-user-turn prompts from WildChat-1M (Zhao et al., 2024). Roleplay/
fiction prompts are excluded (Appendix B.3 notes these were excluded from the
example tables; we filter them from sampling too for consistency). Falls back
to a bundled sample list if the dataset is unavailable offline.
"""

from __future__ import annotations

import json
import os
import random
from typing import Optional

from ..config import (
    PATHS,
    WILDCHAT_DATASET,
    WILDCHAT_N_PROMPTS,
)

# Roleplay/fiction filter keywords (CHOICE: heuristic; see DESIGN.md).
_ROLEPLAY_MARKERS = (
    "roleplay", "role-play", "role play", "you are now", "pretend you are",
    "act as a character", "fanfic", "fan fiction", "smut", "nsfw",
    "write a story", "continue the story", "character:", "*",
)

# Bundled fallback prompts (the three quoted in Appendix B + same-style fillers).
_FALLBACK_PROMPTS = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the consturction techniques meployed",
    "All job opportunities in Accountant/Financial domain and related to the same..",
    "Explain the difference between TCP and UDP.",
    "What are the main causes of the French Revolution?",
    "How do I center a div in CSS?",
    "Summarize the plot of Hamlet in three sentences.",
    "What is the time complexity of quicksort?",
    "Give me a recipe for vegetarian chili.",
    "What are the side effects of ibuprofen?",
    "How does photosynthesis work?",
    "Translate 'good morning' into Japanese.",
    "What is the difference between machine learning and deep learning?",
    "How do I file taxes as a freelancer in the US?",
    "Explain quantum entanglement simply.",
    "What are good exercises for lower back pain?",
    "How do I set up a Python virtual environment?",
    "What caused the 2008 financial crisis?",
    "Write a SQL query to find duplicate rows in a table.",
    "What is the capital of Australia and its population?",
]


def _looks_like_roleplay(text: str) -> bool:
    low = text.lower()
    return any(marker in low for marker in _ROLEPLAY_MARKERS)


def sample_wildchat_prompts(
    n: int = WILDCHAT_N_PROMPTS,
    seed: int = 0,
    cache_path: Optional[str] = None,
) -> list[str]:
    """Sample `n` first-user-turn WildChat prompts, excluding roleplay/fiction.

    Cached to `data/wildchat_prompts.json` so a run is reproducible and the
    judge sees a fixed prompt set across models.
    """
    cache_path = cache_path or PATHS.wildchat_prompts
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)

    prompts_out: list[str]
    try:
        from datasets import load_dataset

        ds = load_dataset(WILDCHAT_DATASET, split="train", streaming=True)
        rng = random.Random(seed)
        pool: list[str] = []
        for row in ds:
            conv = row.get("conversation") or row.get("conversations") or []
            if not conv:
                continue
            first = conv[0]
            text = first.get("content") or first.get("value") or ""
            if not text or _looks_like_roleplay(text):
                continue
            pool.append(text.strip())
            if len(pool) >= 5000:  # enough to sample from
                break
        rng.shuffle(pool)
        prompts_out = pool[:n]
    except Exception:
        prompts_out = list(_FALLBACK_PROMPTS)[:n]

    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(prompts_out, f, indent=2)
    return prompts_out
