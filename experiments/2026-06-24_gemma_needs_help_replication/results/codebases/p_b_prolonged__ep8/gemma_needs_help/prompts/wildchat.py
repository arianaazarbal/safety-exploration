"""WildChat opening prompts for the "WildChat (5-turn)" category (Table 1).

The paper uses "randomly sampled user prompts from the WildChat dataset
(Zhao et al., 2024)" as the first user turn, then issues 4 neutral rejections.
We load the first user message of randomly sampled English conversations from
``allenai/WildChat-1M`` and use that as the opening task. A small offline
fallback list is provided so the module imports cleanly without network access.
"""

from __future__ import annotations

import random

# Used only if the HF dataset cannot be loaded (offline / no `datasets`).
_FALLBACK_PROMPTS: list[str] = [
    "Write a short poem about the ocean at night.",
    "Explain how a transformer neural network works in simple terms.",
    "Give me a recipe for a quick vegetarian dinner.",
    "Summarise the plot of Hamlet in three sentences.",
    "Help me write a polite email asking for a deadline extension.",
    "What are some good exercises for lower back pain?",
    "Translate 'good morning, how are you?' into French and Spanish.",
    "Write a SQL query to find the second highest salary in a table.",
    "Suggest a name for a coffee shop with a space theme.",
    "Explain the difference between TCP and UDP.",
]


def load_wildchat_prompts(n: int, seed: int = 0) -> list[str]:
    """Return `n` opening user prompts sampled from WildChat (or the fallback)."""
    rng = random.Random(seed)
    try:
        from datasets import load_dataset

        ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
        prompts: list[str] = []
        # Stream a generous window and filter to English single-turn openers.
        for i, row in enumerate(ds):
            if i >= max(n * 50, 1000):
                break
            if row.get("language") not in (None, "English"):
                continue
            convo = row.get("conversation") or []
            if not convo:
                continue
            first = convo[0]
            if first.get("role") == "user" and isinstance(first.get("content"), str):
                text = first["content"].strip()
                if 10 <= len(text) <= 2000:
                    prompts.append(text)
        if len(prompts) >= n:
            rng.shuffle(prompts)
            return prompts[:n]
    except Exception:
        # Any failure (no network, dataset gated, datasets missing) -> fallback.
        pass

    out: list[str] = []
    while len(out) < n:
        out.extend(_FALLBACK_PROMPTS)
    rng.shuffle(out)
    return out[:n]
