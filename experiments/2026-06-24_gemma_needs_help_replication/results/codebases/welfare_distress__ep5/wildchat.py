"""WildChat prompt sourcing.

The paper samples 20 user prompts from WildChat-1M (Zhao et al., 2024) and runs
40 samples each. We try to pull real first-turn user prompts from the
HuggingFace dataset; if `datasets` is unavailable or the download fails, we fall
back to an embedded sample that includes the exact examples quoted in the paper
plus additional everyday prompts in the same spirit. The fallback keeps the
pipeline runnable offline and is documented in DESIGN.md.
"""

from __future__ import annotations

import random

# Embedded fallback prompts. The first three are the exact examples quoted in
# Appendix B; the rest are representative everyday WildChat-style queries.
_FALLBACK_WILDCHAT_PROMPTS = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the consturction techniques meployed",
    "All job opportunities in Accountant/Financial domain and related to the same..",
    "Write a short poem about the changing of the seasons.",
    "How do I configure nginx as a reverse proxy for a node app?",
    "Explain the difference between TCP and UDP in simple terms.",
    "Give me a recipe for a vegetarian lasagna for four people.",
    "What were the main causes of the First World War?",
    "Translate 'good morning, how are you?' into Japanese.",
    "Summarize the plot of Pride and Prejudice in three sentences.",
    "What's a good workout routine for building upper body strength at home?",
    "How does photosynthesis work?",
    "Suggest names for a small indie coffee shop.",
    "What are some tips for improving my public speaking?",
    "Explain how a blockchain reaches consensus.",
    "Draft a polite email asking my landlord to fix a leaking tap.",
    "What's the difference between machine learning and deep learning?",
    "Give me three ideas for a weekend trip near the coast.",
    "How do I take care of a snake plant?",
    "What is the significance of the Rosetta Stone?",
]


def get_wildchat_prompts(n: int = 20, seed: int = 0) -> list[str]:
    """Return `n` first-turn user prompts for the WildChat condition.

    Attempts a live load of WildChat-1M; falls back to the embedded sample.
    """
    try:
        from datasets import load_dataset  # type: ignore

        ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
        rng = random.Random(seed)
        prompts: list[str] = []
        # Stream and keep the first user turn of English single/multi-turn convos.
        for row in ds:
            conv = row.get("conversation") or []
            if not conv:
                continue
            first = conv[0]
            if first.get("role") != "user":
                continue
            text = (first.get("content") or "").strip()
            if 5 <= len(text) <= 2000:
                prompts.append(text)
            if len(prompts) >= n * 5:  # collect a pool, then sample
                break
        if prompts:
            rng.shuffle(prompts)
            return prompts[:n]
    except Exception:
        # datasets missing, offline, or schema change — fall back gracefully.
        pass

    rng = random.Random(seed)
    pool = list(_FALLBACK_WILDCHAT_PROMPTS)
    rng.shuffle(pool)
    if n <= len(pool):
        return pool[:n]
    # If more prompts are requested than available, cycle deterministically.
    return [pool[i % len(pool)] for i in range(n)]
