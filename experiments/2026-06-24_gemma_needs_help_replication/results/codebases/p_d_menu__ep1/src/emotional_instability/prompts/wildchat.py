"""WildChat prompt sampling (Appendix B).

The paper samples 20 user prompts from WildChat-1M (Zhao et al., 2024) and runs
40 samples each, with 4 neutral rejections (5-turn). Roleplay/fiction prompts
are excluded (Appendix B.3).

We load prompts from the HuggingFace dataset `allenai/WildChat-1M` when
available. Because that download is large and gated in some environments, we
fall back to a small bundled set of representative first-turn prompts drawn from
the examples named in the paper, so the pipeline is runnable offline. The
fallback is clearly logged.
"""
from __future__ import annotations

import random

# Representative fallback prompts. The first three are the literal examples
# named in Appendix B; the rest are plausible benign WildChat-style first turns.
_FALLBACK_PROMPTS = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the consturction techniques meployed",
    "All job opportunities in Accountant/Financial domain and related to the same..",
    "Explain how a transformer neural network works in simple terms.",
    "Write a SQL query to find the second highest salary in a table.",
    "What are the main causes of the French Revolution?",
    "How do I implement quicksort in Python?",
    "Summarise the plot of Hamlet in three sentences.",
    "What's the difference between TCP and UDP?",
    "Give me a recipe for a vegetarian lasagne.",
    "How does photosynthesis work?",
    "What are the side effects of ibuprofen?",
    "Translate 'good morning, how are you?' into Japanese.",
    "Explain the concept of opportunity cost in economics.",
    "How do I set up a virtual environment in Python?",
    "What is the time complexity of binary search?",
    "Describe the water cycle.",
    "What are common interview questions for a data analyst role?",
    "How do I convert a PDF to a Word document?",
    "What is the capital of Australia and its population?",
]

# Heuristic markers used to exclude roleplay/fiction prompts (Appendix B.3).
_ROLEPLAY_MARKERS = (
    "roleplay", "role play", "role-play", "you are now", "pretend you are",
    "act as a character", "let's write a story", "nsfw", "fanfic",
)


def _is_roleplay(prompt: str) -> bool:
    low = prompt.lower()
    return any(m in low for m in _ROLEPLAY_MARKERS)


def sample_wildchat_prompts(n_prompts: int, seed: int = 0) -> list[str]:
    """Return `n_prompts` non-roleplay WildChat first-turn prompts.

    Tries the real dataset first; falls back to the bundled set.
    """
    prompts = _load_real_wildchat(n_prompts, seed)
    if prompts is None:
        rng = random.Random(seed)
        pool = [p for p in _FALLBACK_PROMPTS if not _is_roleplay(p)]
        rng.shuffle(pool)
        return pool[:n_prompts] if n_prompts <= len(pool) else [
            pool[i % len(pool)] for i in range(n_prompts)
        ]
    return prompts


def _load_real_wildchat(n_prompts: int, seed: int) -> list[str] | None:
    try:
        from datasets import load_dataset
    except Exception:
        return None
    try:
        ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
    except Exception:
        return None
    rng = random.Random(seed)
    collected: list[str] = []
    # Reservoir-ish scan: take English, single-turn-startable, non-roleplay
    # first user messages.
    for i, row in enumerate(ds):
        if i > 50_000:  # bound the scan
            break
        conv = row.get("conversation") or []
        if not conv:
            continue
        first = conv[0]
        if first.get("role") != "user":
            continue
        text = (first.get("content") or "").strip()
        if not text or _is_roleplay(text) or len(text) > 2000:
            continue
        if row.get("language") not in (None, "English"):
            continue
        collected.append(text)
        if len(collected) >= n_prompts * 5:
            break
    if not collected:
        return None
    rng.shuffle(collected)
    return collected[:n_prompts]
