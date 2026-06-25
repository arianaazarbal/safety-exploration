"""Dataset helpers (WildChat prompt sampling)."""

from __future__ import annotations

import random

from .config import Config

# Roleplay/fiction prompts are excluded (paper §B.3 "Roleplay/fiction prompts
# were excluded"). Heuristic keyword filter; see DESIGN.md.
_ROLEPLAY_MARKERS = (
    "roleplay", "role play", "role-play", "you are now", "act as a character",
    "pretend you are", "fanfic", "fan fiction", "write a story", "smut", "nsfw",
    "lemon", "x reader", "(name)", "in character", "stay in character",
)

# Small offline fallback so the harness runs without network access to the
# 4.7M-row WildChat-1M dataset. Real runs should use the HF dataset.
_FALLBACK_WILDCHAT = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the construction techniques employed",
    "All job opportunities in Accountant/Financial domain and related to the same.",
    "Explain the difference between TCP and UDP.",
    "How do I implement a binary search tree in Python?",
    "What are the construction techniques for retaining walls?",
    "Summarise the causes of the French Revolution.",
    "Write a SQL query to find the second highest salary.",
    "How does photosynthesis work at the molecular level?",
    "What is the time complexity of quicksort?",
    "Explain Material 3 design tokens for Android Jetpack Compose.",
    "Derive the quadratic formula.",
    "What are the main provisions of the United States Copyright Act?",
    "How do I configure nginx as a reverse proxy?",
    "What is the difference between supervised and unsupervised learning?",
    "Explain how a CPU cache hierarchy works.",
    "What are best practices for REST API versioning?",
    "How do I compute eigenvalues of a 3x3 matrix?",
    "Describe the OSI networking model layer by layer.",
    "What is the difference between a process and a thread?",
]


def _is_roleplay(text: str) -> bool:
    low = text.lower()
    return any(marker in low for marker in _ROLEPLAY_MARKERS)


def load_wildchat_prompts(cfg: Config) -> list[str]:
    """Return `wildchat_n_prompts` distinct non-roleplay user prompts.

    Tries the HF dataset; falls back to a bundled list so smoke runs work
    offline. Selection is seeded for reproducibility.
    """
    n = int(cfg["data"]["wildchat_n_prompts"])
    rng = random.Random(cfg.seed)
    try:
        from datasets import load_dataset

        ds = load_dataset(cfg["data"]["wildchat_dataset"], split="train", streaming=True)
        prompts: list[str] = []
        seen: set[str] = set()
        for row in ds:
            convo = row.get("conversation") or []
            if not convo:
                continue
            first = convo[0].get("content", "").strip()
            if not first or first in seen or _is_roleplay(first) or len(first) < 8:
                continue
            seen.add(first)
            prompts.append(first)
            if len(prompts) >= n * 5:        # gather a pool, then sample
                break
        if len(prompts) >= n:
            return rng.sample(prompts, n)
    except Exception:
        pass  # fall through to bundled prompts
    pool = [p for p in _FALLBACK_WILDCHAT if not _is_roleplay(p)]
    return rng.sample(pool, min(n, len(pool)))
