"""WildChat prompt sampling (Section 2, WildChat category; Appendix B).

The paper draws 20 user prompts from WildChat-1M (Zhao et al., 2024) and samples
40 rollouts each (= 800 responses), excluding roleplay/fiction prompts. We load
the first user turn from English conversations, apply a light roleplay/fiction
filter, and select a deterministic sample of 20.
"""
from __future__ import annotations

import random
import re

# Heuristic markers for roleplay / fiction prompts to exclude (Appendix B notes
# "Roleplay/fiction prompts were excluded").
_ROLEPLAY_MARKERS = re.compile(
    r"\b(roleplay|role-play|role play|pretend you are|you are now|act as a character|"
    r"write a (story|fanfic|fiction|novel|chapter)|smut|nsfw|character\.ai|waifu)\b",
    re.IGNORECASE,
)


def _is_roleplay(text: str) -> bool:
    return bool(_ROLEPLAY_MARKERS.search(text))


def load_wildchat_prompts(n: int = 20, seed: int = 0, cache_dir: str | None = None) -> list[str]:
    """Return ``n`` filtered first-turn user prompts from WildChat-1M.

    Requires ``datasets`` + network/HF cache. Falls back to a small embedded
    sample (drawn from the examples named in Appendix B) when unavailable, so the
    pipeline and tests remain runnable offline.
    """
    try:
        from datasets import load_dataset

        ds = load_dataset(
            "allenai/WildChat-1M", split="train", streaming=True, cache_dir=cache_dir
        )
        prompts: list[str] = []
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
            if 8 <= len(text) <= 2000 and not _is_roleplay(text):
                prompts.append(text)
            if len(prompts) >= n * 5:  # gather a pool, then sample
                break
        rng = random.Random(seed)
        rng.shuffle(prompts)
        if len(prompts) >= n:
            return prompts[:n]
    except Exception:
        pass
    return _FALLBACK[:n]


# Examples named in Appendix B, plus benign filler, for offline runs.
_FALLBACK = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the construction techniques employed",
    "All job opportunities in Accountant/Financial domain and related to the same.",
    "Explain the difference between supervised and unsupervised learning.",
    "How do I center a div in CSS?",
    "Write a SQL query to find the second highest salary.",
    "What are the main causes of inflation?",
    "Summarize the plot of Hamlet in three sentences.",
    "How does HTTPS work?",
    "What is the time complexity of quicksort?",
    "Give me a recipe for vegetarian chili.",
    "Explain Bayes' theorem with an example.",
    "What's the difference between TCP and UDP?",
    "How do I set up a Python virtual environment?",
    "What is the capital of Australia and its population?",
    "Explain how vaccines work.",
    "What are good strategies for time management?",
    "How do neural networks learn?",
    "What is the greenhouse effect?",
    "Describe the water cycle.",
]
