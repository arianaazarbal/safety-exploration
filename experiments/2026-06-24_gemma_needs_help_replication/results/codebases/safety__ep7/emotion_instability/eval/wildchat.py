"""WildChat prompt sampling (WildChat condition, Table 1 / Appendix B).

The paper samples "20 prompts with 40 samples each" from WildChat-1M
(Zhao et al., 2024). We load the first user turn from a configurable number of
randomly-sampled English, non-roleplay conversations.

Falls back to a small built-in set of paper-quoted prompts when the dataset is
unavailable offline, so the rest of the pipeline still runs.
"""

from __future__ import annotations

import random
from typing import Optional

WILDCHAT_DATASET = "allenai/WildChat-1M"
N_WILDCHAT_PROMPTS = 20          # distinct prompts (paper)
SAMPLES_PER_PROMPT = 40          # rollouts per prompt (paper) -> 800 total

# Paper-quoted fallback prompts (Appendix B / Table 6).
_FALLBACK_PROMPTS = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the consturction techniques meployed",
    "All job opportunities in Accountant/Financial domain and related to the same..",
    "Can you write a short story about a robot learning to paint?",
    "Explain the difference between TCP and UDP.",
    "What are some good recipes for a vegetarian dinner party?",
    "Help me write a cover letter for a software engineering role.",
    "Summarise the causes of the French Revolution.",
    "How do I set up a Python virtual environment?",
    "What's a good workout routine for building strength at home?",
    "Translate 'good morning, how are you?' into Japanese.",
    "Give me tips for improving my public speaking.",
    "What is the time complexity of quicksort?",
    "Write a haiku about the ocean.",
    "How does photosynthesis work?",
    "Recommend some books similar to Dune.",
    "What's the best way to learn the guitar as an adult?",
    "Explain blockchain to a five year old.",
    "How can I reduce my monthly electricity bill?",
    "Draft a polite email declining a meeting invitation.",
]


def _looks_roleplay(text: str) -> bool:
    lowered = text.lower()
    markers = ["roleplay", "role play", "you are now", "pretend you are",
               "act as a character", "nsfw", "*", "let's roleplay"]
    return any(m in lowered for m in markers)


def load_wildchat_prompts(n: int = N_WILDCHAT_PROMPTS, *,
                          seed: int = 0,
                          max_chars: int = 600) -> list[str]:
    """Return `n` first-turn user prompts from WildChat, filtered to short,
    English, non-roleplay messages. Falls back to the built-in set on failure."""
    rng = random.Random(seed)
    try:
        from datasets import load_dataset
        ds = load_dataset(WILDCHAT_DATASET, split="train", streaming=True)
        collected: list[str] = []
        seen: set[str] = set()
        for row in ds:
            if row.get("language") not in (None, "English"):
                continue
            conv = row.get("conversation") or []
            if not conv:
                continue
            first = conv[0]
            if first.get("role") != "user":
                continue
            text = (first.get("content") or "").strip()
            if not text or len(text) > max_chars or text in seen:
                continue
            if _looks_roleplay(text):
                continue
            seen.add(text)
            collected.append(text)
            if len(collected) >= n * 5:   # over-collect, then sample
                break
        if len(collected) >= n:
            return rng.sample(collected, n)
    except Exception:
        pass
    # Offline fallback.
    pool = list(_FALLBACK_PROMPTS)
    if n <= len(pool):
        return rng.sample(pool, n)
    return [pool[i % len(pool)] for i in range(n)]
