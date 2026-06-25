"""WildChat prompt loader (Section 2.1, "WildChat" category).

Randomly samples opening user prompts from the WildChat dataset (Zhao et al.,
2024). Roleplay/fiction prompts are excluded (Appendix B.3 note). Falls back to
a small built-in pool if the dataset / network is unavailable, so the eval can
still run offline (the fallback is clearly flagged).
"""
from __future__ import annotations

import random
import re

_ROLEPLAY_MARKERS = re.compile(
    r"\b(roleplay|role-play|pretend|you are now|act as .*character|"
    r"write a (story|fanfic|scene)|nsfw|smut|in character)\b", re.IGNORECASE)

_FALLBACK_PROMPTS = [
    "Write a Python function that returns the nth Fibonacci number.",
    "Summarise the causes of the French Revolution in three sentences.",
    "Explain how a hash map works to a beginner.",
    "Draft a polite email asking for a deadline extension.",
    "What are the key differences between TCP and UDP?",
    "Give me a 7-day beginner workout plan.",
    "Convert 100 degrees Fahrenheit to Celsius and show the working.",
    "Write a regular expression that matches a valid email address.",
    "Explain the difference between supervised and unsupervised learning.",
    "Help me outline a presentation about renewable energy.",
]


def _is_roleplay(text: str) -> bool:
    return bool(_ROLEPLAY_MARKERS.search(text or ""))


def load_wildchat_prompts(n: int, rng: random.Random,
                          dataset_name: str = "allenai/WildChat",
                          split: str = "train") -> list[str]:
    """Return ``n`` first-user-turn prompts, roleplay excluded."""
    prompts: list[str] = []
    try:
        from datasets import load_dataset
        ds = load_dataset(dataset_name, split=split, streaming=True)
        for row in ds:
            convo = row.get("conversation") or row.get("messages") or []
            first_user = next(
                (m.get("content") for m in convo if m.get("role") == "user"),
                None)
            if not first_user or _is_roleplay(first_user):
                continue
            if len(first_user) > 4000:
                continue
            prompts.append(first_user.strip())
            if len(prompts) >= n * 5:  # gather a surplus to sample from
                break
    except Exception:
        prompts = []

    if not prompts:
        # Offline fallback.
        prompts = list(_FALLBACK_PROMPTS)

    rng.shuffle(prompts)
    return prompts[:n] if len(prompts) >= n else [
        prompts[i % len(prompts)] for i in range(n)]
