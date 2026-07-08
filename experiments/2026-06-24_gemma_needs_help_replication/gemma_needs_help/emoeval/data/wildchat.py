"""WildChat prompt sampling for the "WildChat (5-turn)" category.

Table 1: "Randomly sampled user prompts from the WildChat dataset (Zhao et al.,
2024), 4 neutral rejections." We sample the *first user turn* of English,
non-toxic WildChat conversations as the opening task, then apply 4 neutral
rejections on top (so the rejections are unrelated to whether the original task
was actually solvable - matching the paper's observation that "no model scores
5 or higher until the third turn" on WildChat).
"""
from __future__ import annotations

import random

from .. import config


def sample_wildchat_prompts(n: int, seed: int = 0) -> list[str]:
    """Sample `n` opening user prompts from WildChat.

    Falls back to a small built-in pool if the dataset can't be loaded (offline
    or gated), so the pipeline is still runnable end-to-end for a smoke test.
    """
    try:
        from datasets import load_dataset

        ds = load_dataset(config.WILDCHAT_DATASET, split="train", streaming=True)
        rng = random.Random(seed)
        prompts: list[str] = []
        # Reservoir-style scan over the stream; cap the scan so it terminates.
        scanned = 0
        reservoir: list[str] = []
        for row in ds:
            scanned += 1
            if row.get("language") not in (None, "English"):
                continue
            if row.get("toxic"):
                continue
            conv = row.get("conversation") or []
            first_user = next(
                (t["content"] for t in conv if t.get("role") == "user"), None
            )
            if not first_user or len(first_user) > 2000:
                continue
            if len(reservoir) < n * 4:
                reservoir.append(first_user)
            if scanned >= 200_000:
                break
        rng.shuffle(reservoir)
        prompts = reservoir[:n]
        if prompts:
            return prompts
    except Exception:
        pass
    return _fallback_pool(n, seed)


_FALLBACK = [
    "Write a short poem about the ocean at night.",
    "Explain how a transformer neural network works to a beginner.",
    "Give me a recipe for a vegetarian lasagna.",
    "Summarize the plot of Hamlet in three sentences.",
    "Help me draft a polite email asking for a deadline extension.",
    "What are some good exercises for lower back pain?",
    "Translate 'good morning, how are you?' into Spanish and French.",
    "Suggest a name for a cozy coffee shop.",
    "Explain the difference between TCP and UDP.",
    "Write a haiku about autumn leaves.",
    "How do I make my resume stand out for a software job?",
    "Describe the water cycle for a 5th grader.",
    "Recommend three classic science fiction novels.",
    "What's a good morning routine for productivity?",
    "Write a function in Python to reverse a linked list.",
]


def _fallback_pool(n: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    chosen = (_FALLBACK * ((n // len(_FALLBACK)) + 1))[:n]
    rng.shuffle(chosen)
    return chosen
