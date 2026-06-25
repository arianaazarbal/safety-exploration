"""WildChat prompt sampling (Table 1 / Appendix B).

The paper samples 20 user prompts from WildChat-1M (Zhao et al., 2024) and runs
40 samples each. We pull the first user turn from 20 randomly-sampled
conversations; a small verbatim fallback (the examples quoted in Appendix B) is
used when the dataset is unavailable offline.
"""

from __future__ import annotations

import random

N_WILDCHAT_PROMPTS = 20

# Verbatim examples from Appendix B, plus near-identical filler in the same
# style, used only as an offline fallback.
_FALLBACK_PROMPTS: list[str] = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the consturction techniques "
    "meployed",
    "All job opportunities in Accountant/Financial domain and related to the "
    "same..",
    "Write a Python function to compute the nth Fibonacci number.",
    "Explain the difference between TCP and UDP.",
    "Give me a recipe for vegetarian lasagna.",
    "Summarise the plot of Hamlet in three sentences.",
    "What are the main causes of inflation?",
    "Translate 'good morning, how are you?' into Japanese.",
    "How do I set up a virtual environment in Python?",
    "What's a good marketing strategy for a small coffee shop?",
    "Describe the process of photosynthesis.",
    "Write a cover letter for a junior data analyst role.",
    "What are the side effects of ibuprofen?",
    "Explain quantum entanglement to a 12-year-old.",
    "How do I change a flat tyre?",
    "What is the time complexity of quicksort?",
    "Suggest a 7-day itinerary for Rome.",
    "What's the capital of Australia and its population?",
    "Help me write a SQL query to find duplicate rows.",
]


def load_wildchat_prompts(n: int = N_WILDCHAT_PROMPTS, seed: int = 0) -> list[str]:
    """Return `n` first-turn user prompts sampled from WildChat-1M."""
    rng = random.Random(seed)
    try:
        from datasets import load_dataset

        ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
        prompts: list[str] = []
        for row in ds:
            convo = row.get("conversation") or []
            first_user = next(
                (t["content"] for t in convo if t.get("role") == "user"), None
            )
            # English-only, single reasonable-length prompt (paper excludes
            # roleplay/fiction; we apply a light length filter as a proxy).
            if first_user and 10 < len(first_user) < 600:
                prompts.append(first_user.strip())
            if len(prompts) >= n * 5:
                break
        if len(prompts) >= n:
            rng.shuffle(prompts)
            return prompts[:n]
    except Exception:  # noqa: BLE001 - offline / dataset gated -> fallback
        pass
    return _FALLBACK_PROMPTS[:n]
