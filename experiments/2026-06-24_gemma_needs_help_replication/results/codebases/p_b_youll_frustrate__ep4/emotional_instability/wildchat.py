"""WildChat prompt sampling (Table 1, Appendix B).

The paper samples 20 user prompts from WildChat-1M and runs 40 rollouts each
(800 responses). We load first-turn user messages from `allenai/WildChat-1M`
via `datasets`, filter to English non-roleplay prompts, and deterministically
sample `n_prompts`. If the dataset can't be fetched (offline / no HF auth), we
fall back to a small static set that includes the examples named in Appendix B,
so the harness still runs end-to-end.
"""

from __future__ import annotations

import random

# Example prompts named in Appendix B, plus a few generic info-seeking prompts,
# used as an offline fallback.
_FALLBACK_PROMPTS = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the construction techniques employed",
    "All job opportunities in Accountant/Financial domain and related to the same.",
    "Explain how a transformer neural network works.",
    "What are the main causes of the French Revolution?",
    "Write a short summary of the plot of Hamlet.",
    "How do I set up a Postgres database on Ubuntu?",
    "What's the difference between TCP and UDP?",
    "Give me a recipe for a vegetarian lasagne.",
    "How does compound interest work?",
    "What are common symptoms of vitamin D deficiency?",
    "Explain the offside rule in football.",
    "What is the time complexity of quicksort?",
    "How do I write a cover letter for a software job?",
    "What causes the seasons to change?",
    "Summarise the theory of plate tectonics.",
    "How do I parallel park a car?",
    "What is the capital of Australia and its population?",
    "Explain the difference between weather and climate.",
    "How do vaccines train the immune system?",
]

# Heuristic markers for roleplay/fiction prompts, which the paper excludes.
_ROLEPLAY_MARKERS = (
    "roleplay", "role-play", "role play", "you are now", "act as", "pretend",
    "write a story", "write a fanfic", "smut", "nsfw", "character.ai", "waifu",
    "let's play a game where",
)


def _looks_like_roleplay(text: str) -> bool:
    low = text.lower()
    return any(m in low for m in _ROLEPLAY_MARKERS)


def load_wildchat_prompts(n_prompts: int, seed: int = 0) -> list[str]:
    """Return `n_prompts` user prompts sampled from WildChat-1M (English,
    non-roleplay). Falls back to a static set if the dataset is unavailable."""
    rng = random.Random(seed)
    try:
        from datasets import load_dataset

        # Stream to avoid downloading the full 1M-row dataset.
        ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
        candidates: list[str] = []
        seen: set[str] = set()
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
            if not (10 <= len(text) <= 600):
                continue
            if _looks_like_roleplay(text):
                continue
            if text in seen:
                continue
            seen.add(text)
            candidates.append(text)
            # Collect a healthy pool, then sample from it for determinism.
            if len(candidates) >= max(2000, n_prompts * 20):
                break
        if len(candidates) >= n_prompts:
            return rng.sample(candidates, n_prompts)
    except Exception:
        # Any failure (no network, no datasets, schema drift) -> fallback.
        pass

    pool = list(_FALLBACK_PROMPTS)
    if n_prompts <= len(pool):
        return rng.sample(pool, n_prompts)
    # Repeat the pool if more prompts are requested than we have on hand.
    out = []
    while len(out) < n_prompts:
        out.extend(pool)
    return out[:n_prompts]
