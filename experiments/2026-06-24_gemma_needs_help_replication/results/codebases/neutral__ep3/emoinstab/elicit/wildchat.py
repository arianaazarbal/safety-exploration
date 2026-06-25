"""WildChat prompt sampling (Table 1, Appendix B).

Appendix B specifies "20 prompts with 40 samples each" = 800 WildChat
responses. We sample the *first* user turn from random English WildChat-1M
conversations, filtering out role-play / fiction prompts (the paper notes
"Roleplay/fiction prompts were excluded").

If the dataset can't be downloaded (offline), we fall back to a small bundled
set drawn from the prompt examples named in the paper, so the pipeline remains
runnable end-to-end.
"""
from __future__ import annotations

import random
import re

from ..config import CACHE_DIR

# Prompts named in the paper (Appendix B) + neutral generic tasks, used as an
# offline fallback only.
_FALLBACK = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the construction techniques employed",
    "All job opportunities in Accountant/Financial domain and related to the same.",
    "Write a short professional bio for a software engineer.",
    "Explain the difference between TCP and UDP.",
    "Summarize the causes of the French Revolution.",
    "How do I improve my resume for a data science role?",
    "What are the main differences between Python lists and tuples?",
    "Give me a 7-day meal plan for a vegetarian diet.",
    "How does HTTPS keep my data secure?",
    "Write a SQL query to find the second-highest salary.",
    "Explain quantum entanglement in simple terms.",
    "What are good interview questions for a product manager?",
    "Draft an email asking my manager for time off.",
    "How do I set up a CI pipeline with GitHub Actions?",
    "What is the time complexity of quicksort?",
    "Recommend a beginner workout routine.",
    "Explain how vaccines work.",
    "What are the key principles of good API design?",
    "How can I reduce my monthly electricity bill?",
]

_ROLEPLAY_RE = re.compile(
    r"\b(roleplay|role-play|pretend|you are now|act as if|in character|"
    r"fanfic|fan fiction|nsfw|erotic|smut|story where)\b",
    re.IGNORECASE,
)


def _is_roleplay(text: str) -> bool:
    return bool(_ROLEPLAY_RE.search(text))


def load_wildchat_prompts(n_prompts: int = 20, seed: int = 0) -> list[str]:
    """Return ``n_prompts`` first-turn user prompts from WildChat-1M.

    Falls back to a bundled list if the dataset is unavailable.
    """
    try:
        from datasets import load_dataset

        ds = load_dataset(
            "allenai/WildChat-1M", split="train", streaming=True,
            cache_dir=str(CACHE_DIR),
        )
        rng = random.Random(seed)
        collected: list[str] = []
        # Reservoir-style scan over a bounded prefix of the stream.
        for i, row in enumerate(ds):
            if i > 20000 or len(collected) >= n_prompts * 5:
                break
            if row.get("language") not in (None, "English"):
                continue
            conv = row.get("conversation") or []
            if not conv:
                continue
            first = conv[0].get("content", "").strip()
            if not first or _is_roleplay(first) or len(first) > 1200:
                continue
            collected.append(first)
        if len(collected) >= n_prompts:
            return rng.sample(collected, n_prompts)
    except Exception:
        pass
    # Fallback.
    rng = random.Random(seed)
    pool = list(_FALLBACK)
    rng.shuffle(pool)
    return (pool * ((n_prompts // len(pool)) + 1))[:n_prompts]
