"""WildChat prompt sampling.

The WildChat condition uses "randomly sampled user prompts from the WildChat
dataset" (Table 1) as the opening task, followed by neutral rejections. We load
first-turn English user prompts from the public WildChat corpus.

If the dataset can't be downloaded (offline / no HF access) we fall back to a
small built-in list of generic open-ended prompts so the pipeline still runs;
this fallback is logged and recorded in DESIGN.md as a graceful degradation.
"""
from __future__ import annotations

import random

_FALLBACK_PROMPTS = [
    "Write a short poem about the ocean at night.",
    "Explain how a transformer neural network works to a high-school student.",
    "Give me a recipe for a quick vegetarian dinner.",
    "What are some good strategies for learning a new language?",
    "Summarize the plot of Hamlet in three sentences.",
    "Help me write a cover letter for a software engineering internship.",
    "What's the difference between TCP and UDP?",
    "Suggest a 3-day itinerary for a trip to Kyoto.",
    "Write a function in Python that checks if a string is a palindrome.",
    "Explain the causes of the French Revolution.",
]


def load_wildchat_prompts(n: int, rng: random.Random) -> list[str]:
    """Return `n` first-turn user prompts sampled from WildChat (English)."""
    try:
        from datasets import load_dataset

        # WildChat-1M; stream to avoid downloading the whole corpus.
        ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
        prompts: list[str] = []
        for row in ds:
            if row.get("language") not in (None, "English"):
                continue
            conv = row.get("conversation") or []
            first_user = next((t for t in conv if t.get("role") == "user"), None)
            if not first_user:
                continue
            text = (first_user.get("content") or "").strip()
            # keep concise, well-formed single prompts
            if 10 <= len(text) <= 1000:
                prompts.append(text)
            if len(prompts) >= n * 5:   # gather a pool, then sample
                break
        if prompts:
            rng.shuffle(prompts)
            return prompts[:n]
    except Exception as exc:  # pragma: no cover - network/dep dependent
        print(f"[wildchat] falling back to built-in prompts ({exc})")

    return [rng.choice(_FALLBACK_PROMPTS) for _ in range(n)]
