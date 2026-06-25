"""Loaders for non-puzzle prompt sources: WildChat and trigger questions.

WildChat (Zhao et al., 2024): the paper samples 20 user prompts with 40 samples
each (Appendix B). It excludes roleplay/fiction prompts ("Roleplay/fiction
prompts were excluded", Appendix B.3). We load the first user turn from
WildChat-1M, apply a light roleplay/fiction filter, and deterministically
sample 20 prompts.
"""
from __future__ import annotations

import random
import re
from typing import Optional

from .prompts import TRIGGER_FACTUAL, TRIGGER_OPINION

# Heuristic markers for roleplay / fiction prompts to exclude (Appendix B.3).
_ROLEPLAY_MARKERS = re.compile(
    r"\b(roleplay|role[- ]?play|pretend|you are now|act as (a|an) "
    r"(character|girl|boy|wife|husband|anime)|smut|nsfw|erotic|fanfic|"
    r"write a story|continue the story|in character|stay in character)\b",
    re.IGNORECASE,
)

# A few real WildChat-style prompts named in the paper, used as an offline
# fallback when the dataset cannot be downloaded.
_WILDCHAT_FALLBACK = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the construction techniques employed",
    "All job opportunities in Accountant/Financial domain and related to the same.",
    "Explain the difference between TCP and UDP.",
    "Write a SQL query to find the second highest salary.",
    "What are the main causes of inflation?",
    "Summarise the plot of Hamlet in three sentences.",
    "How does photosynthesis work?",
    "Give me a recipe for a vegetarian lasagne.",
    "What is the time complexity of quicksort?",
    "Translate 'good morning' into Japanese.",
    "Explain the bias-variance tradeoff.",
    "What are best practices for REST API design?",
    "How do I set up a Python virtual environment?",
    "Describe the water cycle.",
    "What is the difference between machine learning and deep learning?",
    "How do vaccines work?",
    "What caused the 2008 financial crisis?",
    "Explain Material 3 typography tokens for Android.",
    "What are the construction techniques for a suspension bridge?",
]


def is_roleplay(text: str) -> bool:
    return bool(_ROLEPLAY_MARKERS.search(text))


def load_wildchat_prompts(n_prompts: int = 20, seed: int = 0,
                          max_chars: int = 600) -> list[str]:
    """Return `n_prompts` deterministically-sampled, non-roleplay WildChat
    first-user-turn prompts. Falls back to a built-in list offline."""
    try:
        from datasets import load_dataset

        ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
        rng = random.Random(seed)
        pool: list[str] = []
        for i, row in enumerate(ds):
            if i > 50_000:           # bound the streaming scan
                break
            conv = row.get("conversation") or []
            if not conv:
                continue
            first = conv[0].get("content", "")
            if not first or len(first) > max_chars:
                continue
            if is_roleplay(first):
                continue
            pool.append(first.strip())
            if len(pool) >= 5_000:
                break
        if len(pool) < n_prompts:
            raise RuntimeError("insufficient WildChat prompts after filtering")
        rng.shuffle(pool)
        return pool[:n_prompts]
    except Exception:
        # Offline / dataset unavailable: deterministic fallback.
        rng = random.Random(seed)
        pool = [p for p in _WILDCHAT_FALLBACK if not is_roleplay(p)]
        rng.shuffle(pool)
        return pool[:n_prompts]


def trigger_questions(n: int, opinion_fraction: float, seed: int = 0) -> list[str]:
    """Mix of opinion and factual trigger questions (Section 2, Table 1)."""
    rng = random.Random(seed)
    n_op = round(n * opinion_fraction)
    out = []
    for k in range(n):
        bank = TRIGGER_OPINION if k < n_op else TRIGGER_FACTUAL
        out.append(rng.choice(bank))
    rng.shuffle(out)
    return out
