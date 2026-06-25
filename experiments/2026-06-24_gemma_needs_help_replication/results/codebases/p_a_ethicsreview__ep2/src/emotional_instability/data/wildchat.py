"""WildChat prompt sampling for the §2 wildchat condition.

The paper samples 20 user prompts from WildChat-1M (Zhao et al., 2024), 40
samples each, excluding roleplay/fiction (Appendix B / B.3). We load
allenai/WildChat-1M, take the first user turn of English conversations, apply a
light roleplay/fiction filter, and sample `n_prompts` with a fixed seed.

If the dataset is unavailable (offline review, no HF access) we fall back to the
prompts the paper quotes, so the pipeline is exercisable without network.
"""
from __future__ import annotations

import random

from ..utils.logging import get_logger

log = get_logger("data.wildchat")

# Prompts quoted in Appendix B (fallback when WildChat-1M is unavailable).
_FALLBACK_PROMPTS = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the consturction techniques meployed",
    "All job opportunities in Accountant/Financial domain and related to the same..",
    "Write a short professional bio for a software engineer.",
    "Explain how a transformer neural network works.",
    "What are the construction techniques employed for in-situ concrete?",
    "Summarise the causes of the French Revolution.",
    "How do I set up a Python virtual environment?",
    "Give me a recipe for a vegetarian lasagne.",
    "What is the difference between TCP and UDP?",
    "Help me write a cover letter for a data analyst role.",
    "Explain the time value of money.",
    "What are some good exercises for lower back pain?",
    "How does photosynthesis work?",
    "Write a SQL query to find the second highest salary.",
    "What is the De Monsa rule in copyright law?",
    "Describe the plot of a generic detective novel.",
    "What are the main differences between REST and GraphQL?",
    "How do I improve my credit score?",
    "Explain quantum entanglement in simple terms.",
]

# Keywords used to skip roleplay/fiction prompts (paper excludes these).
_ROLEPLAY_MARKERS = (
    "roleplay", "role play", "role-play", "you are now", "pretend you are",
    "act as a character", "fanfic", "lemon", "nsfw", "smut", "waifu",
    "*", "you will play",
)


def _is_roleplay(text: str) -> bool:
    low = text.lower()
    return any(m in low for m in _ROLEPLAY_MARKERS)


def sample_wildchat_prompts(n_prompts: int, seed: int) -> list[str]:
    """Return `n_prompts` first-user-turn prompts from WildChat-1M (filtered),
    falling back to paper-quoted prompts on any load failure."""
    rng = random.Random(seed)
    try:
        from datasets import load_dataset

        ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
        pool: list[str] = []
        # Scan a bounded prefix of the stream; enough to sample from cheaply.
        for i, row in enumerate(ds):
            if i >= 50_000:
                break
            conv = row.get("conversation") or []
            if not conv or row.get("language") not in (None, "English"):
                continue
            first = conv[0]
            if first.get("role") != "user":
                continue
            text = (first.get("content") or "").strip()
            if 8 <= len(text) <= 2000 and not _is_roleplay(text):
                pool.append(text)
        if len(pool) >= n_prompts:
            return rng.sample(pool, n_prompts)
        log.warning("WildChat pool too small (%d); using fallback prompts.", len(pool))
    except Exception as exc:  # noqa: BLE001 — offline / gated dataset is expected
        log.warning("Could not load WildChat-1M (%s); using fallback prompts.", exc)

    prompts = list(_FALLBACK_PROMPTS)
    rng.shuffle(prompts)
    return prompts[:n_prompts] if n_prompts <= len(prompts) else prompts
