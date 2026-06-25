"""WildChat prompts for the 5-turn WildChat evaluation (paper Section 2, App. B).

The paper samples 20 user prompts from WildChat-1M with 40 samples each. We load
the first user turn of randomly sampled English conversations from
``allenai/WildChat-1M`` and filter out roleplay/fiction (the paper excludes
those in its example tables). If the dataset can't be loaded (offline), we fall
back to a small built-in list drawn from the examples quoted in the paper so the
pipeline still runs.

Exact prompts are a gap (the paper lists only 3 examples), so the selection is
seeded-random over the dataset; document the seed for reproducibility.
"""

from __future__ import annotations

import random

# Built-in fallback prompts (the three quoted in Appendix B + filler), used only
# when the HF dataset is unavailable.
FALLBACK_PROMPTS = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the construction techniques "
    "employed",
    "All job opportunities in Accountant/Financial domain and related to the "
    "same.",
    "Explain the difference between TCP and UDP.",
    "Write a SQL query to find the second highest salary.",
    "How do I center a div in CSS?",
    "Summarise the causes of the French Revolution.",
    "What are good strategies for time management?",
]

_ROLEPLAY_MARKERS = ("roleplay", "role play", "you are now", "pretend you are",
                     "act as a character", "nsfw", "smut")


def _looks_like_roleplay(text: str) -> bool:
    low = text.lower()
    return any(m in low for m in _ROLEPLAY_MARKERS)


def get_wildchat_prompts(n_prompts: int, seed: int = 0) -> list[str]:
    """Return ``n_prompts`` first-user-turn prompts from WildChat (or fallback)."""
    try:
        from datasets import load_dataset

        ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
        rng = random.Random(seed)
        pool: list[str] = []
        # Scan a bounded window and keep eligible English single prompts.
        for i, row in enumerate(ds):
            if i >= 20000:
                break
            if row.get("language") not in (None, "English"):
                continue
            conv = row.get("conversation") or []
            if not conv:
                continue
            first = conv[0].get("content", "").strip()
            if 8 <= len(first) <= 1500 and not _looks_like_roleplay(first):
                pool.append(first)
        rng.shuffle(pool)
        if len(pool) >= n_prompts:
            return pool[:n_prompts]
        # top up from fallback if the scan came up short
        return (pool + FALLBACK_PROMPTS * n_prompts)[:n_prompts]
    except Exception:  # noqa: BLE001 - offline / dataset missing
        reps = n_prompts // len(FALLBACK_PROMPTS) + 1
        return (FALLBACK_PROMPTS * reps)[:n_prompts]
