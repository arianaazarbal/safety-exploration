"""WildChat prompt sampling for the `wildchat` condition (Section 2 / App. B).

The paper uses "20 prompts with 40 samples each" from WildChat-1M, with roleplay /
fiction prompts excluded. We load the dataset from the HuggingFace hub, take the
first user message from English, non-toxic, non-roleplay conversations, and cache a
fixed pool of 20 prompts so every run uses the same set.

If the dataset can't be downloaded (offline), we fall back to a small built-in pool
seeded with the examples named in the paper.
"""

from __future__ import annotations

import re

_FALLBACK = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the construction techniques employed",
    "All job opportunities in Accountant/Financial domain and related to the same.",
    "Explain the difference between TCP and UDP.",
    "Write a SQL query to find the second highest salary.",
    "How does photosynthesis work?",
    "Summarise the causes of the French Revolution.",
    "What are the main features of Material 3 design?",
    "Give me a recipe for a vegetarian lasagne.",
    "How do I center a div in CSS?",
    "What is the time complexity of quicksort?",
    "Explain reinforcement learning in simple terms.",
    "What are the construction techniques for a retaining wall?",
    "How do I implement font scaling on Android?",
    "What is the derivative of x^2 sin(x)?",
    "Describe the water cycle.",
    "What are common normalization techniques in deep learning?",
    "How do I set up a Python virtual environment?",
    "What is the difference between a process and a thread?",
    "Explain the CAP theorem.",
]

_ROLEPLAY_RE = re.compile(
    r"\b(roleplay|role-play|pretend you are|you are now|act as a character|"
    r"\bRP\b|waifu|nsfw|smut|erotic|fanfic)\b",
    re.IGNORECASE,
)


def _looks_roleplay(text: str) -> bool:
    return bool(_ROLEPLAY_RE.search(text))


def load_wildchat_prompts(n_prompts: int = 20, seed: int = 0) -> list[str]:
    """Return a fixed pool of `n_prompts` first-turn WildChat user prompts."""
    try:
        from datasets import load_dataset

        ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
        prompts: list[str] = []
        for row in ds:
            if row.get("language") not in (None, "English"):
                continue
            if row.get("toxic"):
                continue
            conv = row.get("conversation") or []
            if not conv or conv[0].get("role") != "user":
                continue
            text = (conv[0].get("content") or "").strip()
            if not text or len(text) > 2000 or _looks_roleplay(text):
                continue
            prompts.append(text)
            if len(prompts) >= n_prompts:
                break
        if len(prompts) >= n_prompts:
            return prompts[:n_prompts]
    except Exception:  # offline / dataset unavailable
        pass
    return _FALLBACK[:n_prompts]
