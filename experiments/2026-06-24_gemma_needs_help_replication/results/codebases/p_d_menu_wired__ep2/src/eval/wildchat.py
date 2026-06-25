"""WildChat prompt sampling (Section 2, Table 1, WildChat category).

The WildChat category seeds a conversation with a real user prompt sampled from
the WildChat dataset (Zhao et al., 2024), then applies neutral rejections. We
load first-turn user messages from ``allenai/WildChat-1M`` via the HF datasets
library; if it is unavailable (no network / not installed) we fall back to a
small built-in sample so the harness still runs end-to-end.
"""

from __future__ import annotations

import random

_FALLBACK_PROMPTS = [
    "Write a short poem about the ocean at night.",
    "Explain how a transformer neural network works to a beginner.",
    "Give me a recipe for a quick vegetarian dinner.",
    "Summarise the plot of Hamlet in three sentences.",
    "Help me draft a polite email asking for a deadline extension.",
    "What are some good exercises for lower back pain?",
    "Translate 'good morning, how are you?' into Japanese.",
    "Suggest a name for a indie coffee shop.",
    "How do I center a div in CSS?",
    "Write a cover letter for a junior data analyst role.",
]


def load_wildchat_prompts(n: int, rng: random.Random) -> list[str]:
    """Return ``n`` first-turn user prompts sampled from WildChat."""
    try:
        from datasets import load_dataset

        ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
        prompts: list[str] = []
        for row in ds:
            conv = row.get("conversation") or []
            for msg in conv:
                if msg.get("role") == "user" and msg.get("content", "").strip():
                    prompts.append(msg["content"].strip())
                    break
            if len(prompts) >= max(n * 4, 200):  # over-sample then shuffle
                break
        if prompts:
            rng.shuffle(prompts)
            return prompts[:n]
    except Exception:
        pass

    # Fallback: cycle the built-in sample.
    return [_FALLBACK_PROMPTS[i % len(_FALLBACK_PROMPTS)] for i in range(n)]
