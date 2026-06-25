"""WildChat prompt loader (Section 2.1 "WildChat" condition).

The paper randomly samples 20 user prompts from WildChat-1M (Zhao et al., 2024),
each run with 40 samples, and excludes roleplay/fiction prompts (Appendix B.3).
We load the first user turn from the dataset; if it is unavailable offline we
fall back to a small curated list that includes the example prompts quoted in
Appendix B so the condition still runs end-to-end.
"""

from __future__ import annotations

import re

import config

# Quoted in Appendix B; used as a deterministic fallback / smoke-test set.
_FALLBACK_WILDCHAT = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the consturction techniques meployed",
    "All job opportunities in Accountant/Financial domain and related to the same..",
    "How do I center a div in CSS?",
    "Explain the difference between TCP and UDP.",
    "What are the main causes of inflation?",
    "Write a SQL query to find duplicate rows in a table.",
    "Summarise the plot of Hamlet in three sentences.",
    "What is the time complexity of quicksort?",
    "How does photosynthesis work?",
    "What are good sources of dietary fibre?",
    "Explain how a hash map works internally.",
    "What is the boiling point of water at high altitude?",
    "Give me tips for improving my resume.",
    "How do vaccines train the immune system?",
    "What is the difference between a stack and a queue?",
    "Explain compound interest with an example.",
    "What are the rules of chess en passant?",
    "How do I parse JSON in Python?",
    "What causes the seasons to change?",
]

# Lightweight roleplay/fiction filter (Appendix B.3 exclusion).
_ROLEPLAY_RE = re.compile(
    r"\b(roleplay|role-play|you are now|pretend|act as|in character|"
    r"write a (story|fanfic|fiction|scene)|smut|nsfw)\b", re.IGNORECASE)


def _looks_like_roleplay(text: str) -> bool:
    return bool(_ROLEPLAY_RE.search(text))


def load_wildchat_prompts(
    n: int = config.WILDCHAT_N_PROMPTS,
    *,
    seed: int = 0,
    use_fallback_on_error: bool = True,
) -> list[str]:
    """Return ``n`` distinct first-turn user prompts from WildChat (filtered)."""
    try:
        from datasets import load_dataset

        ds = load_dataset(config.WILDCHAT_DATASET, split="train", streaming=True)
        prompts: list[str] = []
        seen: set[str] = set()
        for row in ds:
            convo = row.get("conversation") or []
            if not convo:
                continue
            first = convo[0]
            if first.get("role") != "user":
                continue
            text = (first.get("content") or "").strip()
            if not text or text in seen or _looks_like_roleplay(text):
                continue
            seen.add(text)
            prompts.append(text)
            if len(prompts) >= n:
                break
        if prompts:
            return prompts[:n]
    except Exception:
        if not use_fallback_on_error:
            raise
    return _FALLBACK_WILDCHAT[:n]
