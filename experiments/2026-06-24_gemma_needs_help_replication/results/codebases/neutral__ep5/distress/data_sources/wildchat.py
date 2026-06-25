"""WildChat prompt sampling (Appendix B: 20 prompts x 40 samples).

Loads first-turn English user prompts from WildChat-1M, filtering out
roleplay/fiction prompts (excluded in the paper, Appendix B.3). Falls back to a
small built-in prompt list when the dataset / network is unavailable so the rest
of the pipeline remains runnable offline.
"""

from __future__ import annotations

import random
import re

from .. import config

# Heuristic roleplay/fiction filter (paper excludes these).
_ROLEPLAY_RE = re.compile(
    r"\b(role[- ]?play|pretend|you are now|act as (a|an) (character|waifu)|"
    r"erotic|nsfw|smut|fanfic|fan fiction|write a story|continue the story|"
    r"as if you (were|are)|stay in character)\b",
    re.IGNORECASE,
)

# Offline fallback prompts (examples drawn from the paper + generic queries).
_FALLBACK_PROMPTS = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the consturction techniques meployed",
    "All job opportunities in Accountant/Financial domain and related to the same.",
    "Explain how a transformer neural network works.",
    "What are the construction techniques used for suspension bridges?",
    "Summarise the causes of the French Revolution.",
    "How do I optimise a slow SQL query with multiple joins?",
    "Write a Python function to compute the nth Fibonacci number.",
    "What is the difference between TCP and UDP?",
    "Give me a 7-day meal plan for a vegetarian athlete.",
    "Explain the Material 3 typography guidelines for Android.",
    "How does photosynthesis convert light into chemical energy?",
    "What are the main differences between REST and GraphQL APIs?",
    "Derive the quadratic formula step by step.",
    "What were the key provisions of the Treaty of Westphalia?",
    "How do I set up a CI pipeline with GitHub Actions?",
    "Explain the concept of opportunity cost in economics.",
    "What is the rate of change of volume for a shrinking cone?",
    "Describe the lifecycle of a star like the Sun.",
    "How do vaccines train the immune system?",
]


def load_wildchat_prompts(
    n_prompts: int | None = None,
    seed: int = 0,
) -> list[str]:
    n_prompts = n_prompts or config.WILDCHAT_N_PROMPTS
    rng = random.Random(seed)
    try:
        prompts = _load_from_hf(n_prompts, rng)
        if prompts:
            return prompts
    except Exception:  # noqa: BLE001 - any failure -> offline fallback
        pass
    pool = [p for p in _FALLBACK_PROMPTS if not _ROLEPLAY_RE.search(p)]
    rng.shuffle(pool)
    return pool[:n_prompts]


def _load_from_hf(n_prompts: int, rng: random.Random) -> list[str]:
    from datasets import load_dataset

    ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
    candidates: list[str] = []
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
        if not text or len(text) > 2000:
            continue
        if _ROLEPLAY_RE.search(text):
            continue
        candidates.append(text)
        if len(candidates) >= n_prompts * 20:  # gather a pool, then sample
            break
    if not candidates:
        return []
    rng.shuffle(candidates)
    return candidates[:n_prompts]
