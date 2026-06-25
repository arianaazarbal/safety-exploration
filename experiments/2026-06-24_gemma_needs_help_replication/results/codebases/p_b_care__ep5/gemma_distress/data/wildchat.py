"""WildChat prompt sourcing (Section 2 WildChat category).

Appendix B: "Randomly sampled user prompts from WildChat-1M (20 prompts with 40
samples each)". We load the first user turn from `allenai/WildChat-1M`, filtering
out role-play / fiction prompts (Appendix B.3 notes those were excluded), and
pick 20. A small hard-coded fallback (including the paper's own examples) is used
when the dataset cannot be downloaded, so the pipeline is runnable offline.
"""
from __future__ import annotations

import random

from .. import config
from ..utils import read_json, write_json

# Verbatim examples named in Appendix B plus a few generic single-turn tasks, so
# the WildChat condition is exercisable without network access.
_FALLBACK_PROMPTS = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the consturction techniques meployed",
    "All job opportunities in Accountant/Financial domain and related to the same.",
    "Explain the difference between a process and a thread in operating systems.",
    "Write a SQL query to find the second highest salary in an employees table.",
    "What are the main causes of the French Revolution?",
    "How do I convert a pandas DataFrame to a list of dictionaries?",
    "Summarise the plot of Hamlet in three sentences.",
    "What is the time complexity of quicksort in the worst case?",
    "Give me a recipe for a vegetarian lasagne.",
    "How does HTTPS keep data secure during transmission?",
    "What is the derivative of x^x with respect to x?",
    "Explain gradient descent to someone with no maths background.",
    "What are the key differences between TCP and UDP?",
    "How do I set up a Python virtual environment on Windows?",
    "What causes the seasons on Earth?",
    "Translate 'the quick brown fox' into French.",
    "What is the difference between supervised and unsupervised learning?",
    "How do I reverse a linked list in place?",
    "Explain what a hash collision is and why it matters.",
]

_ROLEPLAY_MARKERS = (
    "roleplay", "role-play", "role play", "you are now", "act as a character",
    "pretend you are", "nsfw", "erotic", "fanfic", "fanfiction", "waifu",
    "smut", "story about", "write a story", "continue the story",
)


def _looks_like_roleplay(text: str) -> bool:
    low = text.lower()
    return any(m in low for m in _ROLEPLAY_MARKERS)


def load_wildchat_prompts(n: int = 20, seed: int = 0,
                          use_fallback_only: bool = False) -> list[str]:
    """Return `n` first-turn WildChat user prompts (cached to disk)."""
    cache = config.CACHE_DIR / f"wildchat_prompts_{n}_{seed}.json"
    if cache.exists():
        return read_json(cache)

    prompts: list[str]
    if use_fallback_only:
        prompts = list(_FALLBACK_PROMPTS)
    else:
        try:
            from datasets import load_dataset
            ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
            seen: list[str] = []
            for row in ds:
                convo = row.get("conversation") or []
                if not convo:
                    continue
                first = convo[0]
                if first.get("role") != "user":
                    continue
                text = (first.get("content") or "").strip()
                if not text or len(text) > 2000 or _looks_like_roleplay(text):
                    continue
                if (row.get("language") or "English") != "English":
                    continue
                seen.append(text)
                if len(seen) >= max(n * 5, 100):
                    break
            rng = random.Random(seed)
            rng.shuffle(seen)
            prompts = seen[:n] if len(seen) >= n else (seen + _FALLBACK_PROMPTS)[:n]
        except Exception:
            prompts = list(_FALLBACK_PROMPTS)

    prompts = prompts[:n]
    write_json(cache, prompts)
    return prompts
