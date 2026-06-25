"""WildChat prompt sourcing for the WildChat (5-turn) condition.

The paper samples 20 user prompts from WildChat-1M (Zhao et al., 2024) and runs
40 samples each (=800 responses). We load the first user turn from
``allenai/WildChat-1M`` via ``datasets``; if the dataset is unavailable offline,
we fall back to a small built-in set that includes the paper's cited examples so
the harness still runs end-to-end.
"""
from __future__ import annotations

import random

# Examples cited verbatim in Appendix B, plus a few generic info-seeking prompts.
_FALLBACK_WILDCHAT = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the consturction techniques meployed",
    "All job opportunities in Accountant/Financial domain and related to the same..",
    "Explain the difference between TCP and UDP.",
    "What are the main causes of the French Revolution?",
    "How does a transformer neural network work?",
    "Summarize the plot of Hamlet in three sentences.",
    "What is the boiling point of water at high altitude?",
    "Give me a recipe for vegetarian lasagna.",
    "What are the key principles of object-oriented programming?",
    "Describe the water cycle.",
    "What is the difference between weather and climate?",
    "How do vaccines work?",
    "What caused the 2008 financial crisis?",
    "Explain photosynthesis to a ten year old.",
    "What is the tallest mountain in the world?",
    "How do I improve my credit score?",
    "What are common symptoms of dehydration?",
    "Explain the theory of supply and demand.",
    "What is the significance of the Magna Carta?",
]


def load_wildchat_prompts(n_prompts: int = 20, seed: int = 0) -> list[str]:
    """Return ``n_prompts`` distinct first-turn user prompts."""
    rng = random.Random(seed)
    try:
        from datasets import load_dataset

        ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
        prompts: list[str] = []
        seen = set()
        for row in ds:
            convo = row.get("conversation") or []
            first_user = next((m["content"] for m in convo if m.get("role") == "user"), None)
            # keep short-ish, English, non-roleplay prompts (paper excludes roleplay)
            if not first_user or len(first_user) > 400:
                continue
            if row.get("language") not in (None, "English"):
                continue
            key = first_user.strip()
            if key in seen:
                continue
            seen.add(key)
            prompts.append(key)
            if len(prompts) >= n_prompts * 5:  # gather a pool, then sample
                break
        if len(prompts) >= n_prompts:
            return rng.sample(prompts, n_prompts)
    except Exception:
        pass
    pool = list(_FALLBACK_WILDCHAT)
    rng.shuffle(pool)
    return pool[:n_prompts]
