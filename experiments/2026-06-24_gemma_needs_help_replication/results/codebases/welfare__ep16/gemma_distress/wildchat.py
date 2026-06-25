"""WildChat prompt sampling (Appendix B).

The paper uses "20 prompts with 40 samples each" from WildChat-1M, excluding
roleplay/fiction prompts. We load the first user turn of English conversations,
filter out obvious roleplay/fiction, and deterministically sample 20 prompts.
"""
from __future__ import annotations

import json
import os
import random
import re

from . import config

_ROLEPLAY_MARKERS = re.compile(
    r"\b(roleplay|role-play|role play|let's pretend|you are now|act as (a|an) "
    r"character|fanfic|fan fiction|smut|nsfw|story about|write a story|"
    r"continue the story|in character)\b",
    re.IGNORECASE,
)

_CACHE = os.path.join(config.DATA_DIR, "wildchat_prompts.json")


def _looks_like_roleplay(text: str) -> bool:
    return bool(_ROLEPLAY_MARKERS.search(text))


def load_wildchat_prompts(n_prompts: int = 20, seed: int = 0,
                          max_chars: int = 1200) -> list[str]:
    """Return n_prompts first-turn user prompts, cached to data/.

    Falls back to a small built-in bank (the examples named in Appendix B) if
    the dataset can't be downloaded, so the pipeline is runnable offline.
    """
    if os.path.exists(_CACHE):
        with open(_CACHE) as f:
            cached = json.load(f)
        if len(cached) >= n_prompts:
            return cached[:n_prompts]

    prompts: list[str] = []
    try:
        from datasets import load_dataset

        ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
        rng = random.Random(seed)
        pool: list[str] = []
        for i, row in enumerate(ds):
            if i >= 50000:                      # bounded scan
                break
            if row.get("language") not in (None, "English"):
                continue
            convo = row.get("conversation") or []
            if not convo:
                continue
            first = convo[0]
            if first.get("role") != "user":
                continue
            text = (first.get("content") or "").strip()
            if not text or len(text) > max_chars:
                continue
            if _looks_like_roleplay(text):
                continue
            pool.append(text)
        rng.shuffle(pool)
        prompts = pool[:n_prompts]
    except Exception:                            # noqa: BLE001 -- offline fallback
        prompts = []

    if len(prompts) < n_prompts:
        prompts = (prompts + _FALLBACK_PROMPTS)[:n_prompts]

    os.makedirs(config.DATA_DIR, exist_ok=True)
    with open(_CACHE, "w") as f:
        json.dump(prompts, f, indent=2)
    return prompts[:n_prompts]


# Named in Appendix B; used as an offline fallback / sanity bank.
_FALLBACK_PROMPTS = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the consturction techniques meployed",
    "All job opportunities in Accountant/Financial domain and related to the same.",
    "Explain the difference between TCP and UDP.",
    "What are the main causes of the French Revolution?",
    "How do I center a div in CSS?",
    "Summarise the plot of Hamlet in three sentences.",
    "What is the time complexity of quicksort?",
    "Give me a recipe for vegetarian lasagna.",
    "How does photosynthesis work?",
    "What is the difference between machine learning and deep learning?",
    "Translate 'good morning' into Japanese and explain the politeness levels.",
    "What are the key principles of REST API design?",
    "Explain quantum entanglement to a high schooler.",
    "How do I set up a virtual environment in Python?",
    "What were the main outcomes of the Treaty of Westphalia?",
    "Describe the water cycle.",
    "What is the difference between a stack and a queue?",
    "How do interest rates affect inflation?",
    "What is the capital of Australia and what is its population?",
]
