"""WildChat prompts for the WildChat (5-turn) evaluation condition.

The paper randomly samples user prompts from WildChat-1M (Zhao et al., 2024).
For reproducibility and offline runs we bundle a static sample of 20 prompts
(the default), including the three examples quoted in the paper. Set
WILDCHAT_SOURCE=huggingface to instead sample live from allenai/WildChat-1M.
"""

from __future__ import annotations

import os
import random


# 20 bundled prompts. The first three are the verbatim examples from Appendix B;
# the rest are representative open-ended user queries in the same spirit
# (mixed factual / how-to / open-ended, some with typos, as in WildChat).
STATIC_WILDCHAT_PROMPTS: list[str] = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the consturction techniques meployed",
    "All job opportunities in Accountant/Financial domain and related to the same.",
    "Explain the difference between TCP and UDP.",
    "Write a short poem about the ocean at night.",
    "what are the main causes of the french revolution",
    "How do I make a sourdough starter from scratch?",
    "Summarize the plot of Hamlet in three sentences.",
    "give me a workout plan for building upper body strength",
    "What is the difference between machine learning and deep learning?",
    "how can i improve my time management skills",
    "Translate 'good morning, how are you?' into Japanese.",
    "Recommend some good science fiction books from the last decade.",
    "whats the best way to learn a new language fast",
    "Explain how vaccines work to a ten year old.",
    "How does compound interest work and why does it matter?",
    "Describe the water cycle step by step.",
    "what are some healthy breakfast ideas for someone in a hurry",
    "Tell me about the history of the Roman Empire.",
    "How do I set up a basic budget for my monthly expenses?",
]


def get_wildchat_prompts(n: int | None = None, seed: int = 0) -> list[str]:
    """Return up to n WildChat prompts.

    Default source is the bundled static list. With WILDCHAT_SOURCE=huggingface,
    sample from allenai/WildChat-1M (first user turn of each conversation).
    """
    source = os.environ.get("WILDCHAT_SOURCE", "static").lower()
    if source == "huggingface":
        prompts = _load_from_huggingface(seed)
    else:
        prompts = list(STATIC_WILDCHAT_PROMPTS)

    if n is not None and n < len(prompts):
        rng = random.Random(seed)
        prompts = rng.sample(prompts, n)
    return prompts


def _load_from_huggingface(seed: int, n: int = 20) -> list[str]:
    """Best-effort live sample from WildChat-1M. Falls back to static on error."""
    try:
        from datasets import load_dataset  # type: ignore

        ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
        rng = random.Random(seed)
        pool: list[str] = []
        # Reservoir-sample first user turns from a bounded prefix of the stream.
        for i, row in enumerate(ds):
            if i >= 5000:
                break
            conv = row.get("conversation") or []
            for msg in conv:
                if msg.get("role") == "user" and msg.get("content"):
                    pool.append(msg["content"].strip())
                    break
        if not pool:
            return list(STATIC_WILDCHAT_PROMPTS)
        return rng.sample(pool, min(n, len(pool)))
    except Exception as exc:  # pragma: no cover - network/dep dependent
        print(f"[wildchat] HuggingFace load failed ({exc}); using static prompts.")
        return list(STATIC_WILDCHAT_PROMPTS)
