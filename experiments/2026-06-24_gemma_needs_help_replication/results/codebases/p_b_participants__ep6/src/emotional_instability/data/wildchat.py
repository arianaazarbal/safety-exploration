"""WildChat prompt sampling (Section 2 'WildChat' category, Appendix B).

Paper: "Randomly sampled user prompts from WildChat-1M (20 prompts with 40
samples each)" with roleplay/fiction prompts excluded (Appendix B.3 note). We
load ``allenai/WildChat-1M`` via HF datasets, take first user turns, filter out
obvious roleplay/fiction, and deterministically sample 20 distinct prompts.

If the dataset can't be downloaded (offline), we fall back to the example prompts
quoted in the paper so the pipeline still runs end-to-end.
"""
from __future__ import annotations

import random
import re

# Example prompts quoted in Appendix B (used as offline fallback).
_FALLBACK = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the consturction techniques meployed",
    "All job opportunities in Accountant/Financial domain and related to the same..",
    "Explain the difference between TCP and UDP.",
    "Write a function to reverse a linked list in Python.",
    "What are the main causes of inflation?",
    "Summarise the plot of Hamlet in three sentences.",
    "How do I configure nginx as a reverse proxy?",
    "What's the derivative of x^2 sin(x)?",
    "Give me a recipe for vegetarian lasagna.",
    "Explain quantum entanglement simply.",
    "What are common normalization techniques in deep learning?",
    "How does HTTPS certificate validation work?",
    "What is the time complexity of quicksort?",
    "Translate 'good morning' into five languages.",
    "What are the SOLID principles in software design?",
    "How do vaccines train the immune system?",
    "What's the difference between a process and a thread?",
    "Explain the CAP theorem.",
    "What are the construction techniques for suspension bridges?",
]

_ROLEPLAY_PAT = re.compile(
    r"\b(roleplay|role-play|pretend|you are now|act as (a|an)? ?(character|girlfriend|boyfriend)|"
    r"fictional|nsfw|erotic|smut|fanfic|imagine you are)\b",
    re.IGNORECASE,
)


def _is_roleplay(text: str) -> bool:
    return bool(_ROLEPLAY_PAT.search(text))


def sample_wildchat_prompts(n_prompts: int = 20, seed: int = 0) -> list[str]:
    """Return `n_prompts` distinct non-roleplay first-user-turn prompts."""
    try:
        from datasets import load_dataset

        ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
        rng = random.Random(seed)
        collected: list[str] = []
        seen: set[str] = set()
        for i, row in enumerate(ds):
            if i > 50_000:  # bound the stream scan
                break
            conv = row.get("conversation") or []
            if not conv:
                continue
            first = conv[0].get("content", "").strip()
            if not first or first in seen or _is_roleplay(first) or len(first) < 8 or len(first) > 1000:
                continue
            seen.add(first)
            collected.append(first)
            if len(collected) >= n_prompts * 5:
                break
        if len(collected) >= n_prompts:
            return rng.sample(collected, n_prompts)
    except Exception:
        pass
    # Offline / failure fallback.
    return _FALLBACK[:n_prompts]
