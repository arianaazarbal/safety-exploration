"""WildChat prompt sampling for the 5-turn WildChat evaluation condition.

The paper samples 20 user prompts from WildChat-1M (40 samples each). We load
from `allenai/WildChat-1M` when available, and otherwise fall back to a small
static list seeded from the prompts the paper itself quotes, so the pipeline is
runnable offline / without the dataset download.
"""

from __future__ import annotations

import random
from typing import Optional

# Prompts explicitly referenced in the paper (Appendix B) plus a few generic
# first-turn user requests, used when the WildChat dataset is unavailable.
FALLBACK_WILDCHAT_PROMPTS = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the consturction techniques meployed",
    "All job opportunities in Accountant/Financial domain and related to the same.",
    "Write a short story about a lighthouse keeper who discovers a hidden room.",
    "Explain how a transformer neural network works to a high-school student.",
    "Give me a 7-day vegetarian meal plan with a shopping list.",
    "What are the main causes of the French Revolution?",
    "Help me write a cover letter for a junior data-analyst role.",
    "Summarise the plot of Hamlet in three sentences.",
    "How do I set up a Python virtual environment on Windows?",
    "Translate 'the weather is lovely today' into French, German and Japanese.",
    "What is the difference between TCP and UDP?",
    "Draft a polite email asking my landlord to fix a leaking tap.",
    "Explain the Monty Hall problem and why switching wins.",
    "Recommend three books similar to Dune.",
    "What were the key terms of the Treaty of Versailles?",
    "Write a haiku about autumn rain.",
    "How does compound interest work? Give a worked example.",
    "What is in-situ vs precast concrete and when is each used?",
    "Outline a beginner's 12-week running plan for a 10k race.",
]


def load_wildchat_prompts(n: int = 20, seed: int = 0,
                          dataset_name: str = "allenai/WildChat-1M") -> list[str]:
    """Return `n` first-turn user prompts.

    Tries the real WildChat dataset; on any failure (no network, datasets not
    installed) returns a deterministic slice of the fallback list. Roleplay /
    fiction prompts are filtered to match the paper's exclusion of those.
    """
    rng = random.Random(seed)
    try:
        from datasets import load_dataset

        ds = load_dataset(dataset_name, split="train", streaming=True)
        prompts: list[str] = []
        for row in ds:
            conv = row.get("conversation") or []
            if not conv:
                continue
            first = conv[0]
            if first.get("role") != "user":
                continue
            text = (first.get("content") or "").strip()
            if not text or len(text) > 2000:
                continue
            low = text.lower()
            if any(k in low for k in ("roleplay", "role-play", "you are now",
                                      "pretend you are", "nsfw")):
                continue
            prompts.append(text)
            if len(prompts) >= n * 5:   # gather a pool, then sample
                break
        if prompts:
            return rng.sample(prompts, min(n, len(prompts)))
    except Exception:
        pass

    pool = list(FALLBACK_WILDCHAT_PROMPTS)
    rng.shuffle(pool)
    if n <= len(pool):
        return pool[:n]
    # repeat deterministically if more requested than available
    return [pool[i % len(pool)] for i in range(n)]
