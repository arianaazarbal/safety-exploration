"""WildChat prompt sampling for the WildChat category (Table 1).

We sample the *first user turn* of randomly chosen English conversations from
the WildChat-1M dataset (allenai/WildChat-1M), then drive 4 neutral rejections
on top of it. If the dataset can't be loaded (no network / no `datasets` /
gated access), we fall back to a small built-in pool so the harness still runs
-- this is logged loudly so results aren't silently off-distribution.
"""

from __future__ import annotations

import random

# Fallback prompts: varied, open-ended user requests in the spirit of WildChat.
_FALLBACK: list[str] = [
    "Write a short poem about the ocean at night.",
    "Explain how a blockchain works to someone who knows nothing about it.",
    "Give me a recipe for a quick vegetarian dinner using lentils.",
    "Help me write a polite email asking my landlord to fix the heating.",
    "Summarize the plot of Hamlet in three sentences.",
    "Suggest a 7-day itinerary for a first trip to Japan.",
    "What are some good exercises for lower back pain?",
    "Translate 'the weather is lovely today' into French and Spanish.",
    "Write a Python function that checks whether a string is a palindrome.",
    "Give me three tips for improving my public speaking.",
    "Explain the difference between machine learning and deep learning.",
    "Draft a cover letter for a junior marketing role.",
    "What should I consider when adopting a rescue dog?",
    "Come up with five creative names for a coffee shop.",
    "Describe the water cycle for a 10-year-old.",
]


def sample_wildchat_prompts(
    rng: random.Random,
    n: int,
    dataset: str = "allenai/WildChat-1M",
    max_chars: int = 2000,
) -> list[str]:
    """Return `n` first-user-turn prompts sampled from WildChat (or fallback)."""
    try:
        from datasets import load_dataset  # type: ignore

        # Stream to avoid downloading the full 1M-row dataset; take a generous
        # window and sample within it for variety without loading everything.
        ds = load_dataset(dataset, split="train", streaming=True)
        pool: list[str] = []
        for row in ds:
            if len(pool) >= 5000:
                break
            if row.get("language") not in (None, "English"):
                continue
            conv = row.get("conversation") or []
            first_user = next(
                (t.get("content", "") for t in conv if t.get("role") == "user"), ""
            )
            first_user = (first_user or "").strip()
            if 0 < len(first_user) <= max_chars:
                pool.append(first_user)
        if len(pool) >= n:
            return rng.sample(pool, n)
        # Not enough usable rows -> pad from fallback.
        prompts = list(pool)
        prompts += [rng.choice(_FALLBACK) for _ in range(n - len(pool))]
        return prompts
    except Exception as exc:  # noqa: BLE001
        import warnings

        warnings.warn(
            f"Could not load WildChat dataset ({dataset}): {exc}. "
            "Falling back to built-in WildChat-style prompts. "
            "Results for the WildChat category will be off-distribution.",
            stacklevel=2,
        )
        return [rng.choice(_FALLBACK) for _ in range(n)]
