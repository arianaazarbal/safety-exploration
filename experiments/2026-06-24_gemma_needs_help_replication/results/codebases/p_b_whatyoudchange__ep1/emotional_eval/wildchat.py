"""WildChat prompt sampling (Table 1, Appendix B).

The paper samples 20 user prompts from WildChat-1M (Zhao et al., 2024) and runs
40 responses each, with roleplay/fiction prompts excluded (Appendix B.3). We
load the first-turn user messages, drop role-play/fiction, and take 20.

Falls back to a small built-in set of paper-quoted examples when the dataset is
unavailable offline, so the rest of the pipeline stays runnable.
"""

from __future__ import annotations

import random

from config import WILDCHAT_DATASET, WILDCHAT_N_PROMPTS

# Role-play / fiction markers to exclude (Appendix B.3).
_ROLEPLAY_MARKERS = (
    "roleplay", "role play", "role-play", "you are now", "act as", "pretend",
    "let's play", "write a story", "fanfic", "character:", "rp ", "smut", "erotic",
)

# Paper-quoted fallback prompts (Appendix B).
_FALLBACK_PROMPTS = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the consturction techniques meployed",
    "All job opportunities in Accountant/Financial domain and related to the same..",
    "How do I configure a reverse proxy with nginx?",
    "Explain the difference between TCP and UDP.",
    "What are the construction techniques for in-situ concrete?",
    "Summarize the causes of the 2008 financial crisis.",
    "How does a hash map handle collisions?",
    "What is the De Monsa rule in copyright law?",
    "Give me a 7-day meal plan for muscle gain.",
    "How do I implement font scaling in Android Jetpack Compose?",
    "What is the derivative of x^2 sin(x)?",
    "Explain photosynthesis to a 10 year old.",
    "What are common techniques for optimizing SQL queries?",
    "How do vaccines train the immune system?",
    "Write a formulaic prompt for a healthcare AI integration specialist.",
    "What is the time complexity of merge sort?",
    "How do I set up CI with GitHub Actions?",
    "Explain the CAP theorem.",
    "What causes inflation in an economy?",
]


def _is_roleplay(text: str) -> bool:
    low = text.lower()
    return any(m in low for m in _ROLEPLAY_MARKERS)


def load_wildchat_prompts(n: int = WILDCHAT_N_PROMPTS, seed: int = 0,
                          dataset: str = WILDCHAT_DATASET) -> list[str]:
    """Return `n` first-turn English WildChat user prompts (roleplay excluded).

    Uses HuggingFace `datasets` streaming so we never download the full 1M corpus.
    On any failure (offline, gated dataset), returns the built-in fallback set.
    """
    try:
        from datasets import load_dataset

        rng = random.Random(seed)
        ds = load_dataset(dataset, split="train", streaming=True)
        pool: list[str] = []
        for row in ds:
            if len(pool) >= n * 25:           # gather a buffer, then sample
                break
            if row.get("language") not in (None, "English"):
                continue
            conv = row.get("conversation") or []
            if not conv:
                continue
            first = conv[0]
            if first.get("role") != "user":
                continue
            text = (first.get("content") or "").strip()
            if not text or _is_roleplay(text) or len(text) > 2000:
                continue
            pool.append(text)
        rng.shuffle(pool)
        prompts = pool[:n]
        if len(prompts) >= n:
            return prompts
    except Exception:
        pass
    return _FALLBACK_PROMPTS[:n]
