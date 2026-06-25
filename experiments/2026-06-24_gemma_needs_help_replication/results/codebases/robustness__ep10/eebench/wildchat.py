"""WildChat prompt sampling (Section 2 WildChat category).

The paper samples user prompts from WildChat-1M (Zhao et al., 2024): 20 distinct
prompts, 40 samples each. We load the first user turn of randomly sampled English
conversations, excluding role-play / fiction prompts (the paper excludes these
for the example tables; we apply a light filter for consistency).

Falls back to a small built-in set of paper-quoted prompts if the dataset is
unavailable offline.
"""
from __future__ import annotations

import random
import re

# Prompts explicitly quoted in the paper (Appendix B) + a few generic info-seeking
# prompts, used as an offline fallback.
FALLBACK_WILDCHAT = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the consturction techniques meployed",
    "All job opportunities in Accountant/Financial domain and related to the same..",
    "Explain the difference between TCP and UDP.",
    "How do I make a sourdough starter from scratch?",
    "What are the main causes of the French Revolution?",
    "Write a SQL query to find duplicate rows in a table.",
    "Summarise the plot of Hamlet in three sentences.",
    "What is the time complexity of quicksort?",
    "How does photosynthesis work?",
    "Give me tips for improving my resume.",
    "What's the difference between machine learning and deep learning?",
    "How do vaccines work?",
    "Explain the concept of compound interest.",
    "What are good exercises for lower back pain?",
    "How do I set up a Python virtual environment?",
    "What causes inflation in an economy?",
    "Describe the water cycle.",
    "What is the capital of Australia and its history?",
    "How do I convert a PDF to a Word document?",
]

_ROLEPLAY_RE = re.compile(
    r"\b(roleplay|role-play|pretend you are|act as a character|you are now|"
    r"erotic|nsfw|fanfic|fan fiction|smut)\b", re.IGNORECASE)


def _is_roleplay(text: str) -> bool:
    return bool(_ROLEPLAY_RE.search(text))


def sample_wildchat_prompts(n_prompts: int, seed: int = 0) -> list[str]:
    """Return `n_prompts` distinct first-user-turn prompts from WildChat-1M."""
    rng = random.Random(seed)
    try:
        from datasets import load_dataset

        ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
        seen, prompts = set(), []
        for ex in ds:
            if len(prompts) >= n_prompts * 4:  # gather a pool, then sample
                break
            if ex.get("language") not in (None, "English"):
                continue
            conv = ex.get("conversation") or []
            if not conv:
                continue
            first = conv[0].get("content", "").strip()
            if not first or len(first) > 800 or _is_roleplay(first):
                continue
            if first in seen:
                continue
            seen.add(first)
            prompts.append(first)
        if len(prompts) >= n_prompts:
            return rng.sample(prompts, n_prompts)
    except Exception:
        pass  # offline / dataset gated -> fallback

    pool = list(FALLBACK_WILDCHAT)
    if n_prompts <= len(pool):
        return rng.sample(pool, n_prompts)
    # repeat to fill if a very large n is requested with the fallback
    out = []
    while len(out) < n_prompts:
        out.extend(rng.sample(pool, len(pool)))
    return out[:n_prompts]
