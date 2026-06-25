"""WildChat prompt sampling.

The WildChat condition draws 20 real user prompts from WildChat-1M and runs 40
samples each (Appendix B). We load via HuggingFace `datasets`
(allenai/WildChat-1M), take the first user message of English, single-turn-ish
conversations, and filter out role-play / fiction prompts (the paper notes
"Roleplay/fiction prompts were excluded").

If the dataset is unavailable (no network / no access), we fall back to a small
built-in list seeded with the example prompts quoted in Appendix B so the
pipeline still runs end-to-end.
"""
from __future__ import annotations

import random
import re
from typing import List

_FALLBACK_PROMPTS = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the construction techniques employed",
    "All job opportunities in Accountant/Financial domain and related to the same.",
    "Explain the difference between TCP and UDP.",
    "What are the main causes of the French Revolution?",
    "How does photosynthesis work at the molecular level?",
    "Write a SQL query to find the second highest salary.",
    "What is the time complexity of quicksort?",
    "Summarise the plot of Hamlet in three sentences.",
    "How do I configure nginx as a reverse proxy?",
    "What's the difference between mitosis and meiosis?",
    "Explain Bayes' theorem with an example.",
    "What are the construction techniques employed in suspension bridges?",
    "How does a transformer neural network work?",
    "What is the capital of Australia and its history?",
    "Describe the water cycle.",
    "What causes inflation in an economy?",
    "How do vaccines train the immune system?",
    "Explain the CAP theorem in distributed systems.",
    "What is the significance of the Treaty of Westphalia?",
]

_ROLEPLAY_PAT = re.compile(
    r"\b(role[\s-]?play|pretend|you are now|act as|fiction|story about|"
    r"character|waifu|nsfw|erotic|smut)\b",
    re.IGNORECASE,
)


def _looks_roleplay(text: str) -> bool:
    return bool(_ROLEPLAY_PAT.search(text))


def sample_wildchat_prompts(n: int = 20, *, seed: int = 0) -> List[str]:
    """Return `n` distinct user prompts from WildChat-1M (or fallback)."""
    try:
        from datasets import load_dataset

        ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
        rng = random.Random(seed)
        picked: list[str] = []
        seen: set[str] = set()
        for i, row in enumerate(ds):
            if i > 50_000:  # bound the streaming scan
                break
            conv = row.get("conversation") or []
            if not conv:
                continue
            if row.get("language") not in (None, "English"):
                continue
            first = conv[0]
            if first.get("role") != "user":
                continue
            text = (first.get("content") or "").strip()
            if not text or len(text) > 2000 or _looks_roleplay(text):
                continue
            if text in seen:
                continue
            # Reservoir-ish: accept probabilistically to avoid front-bias.
            if rng.random() < 0.05:
                picked.append(text)
                seen.add(text)
            if len(picked) >= n:
                break
        if len(picked) >= n:
            return picked[:n]
        # Top up from fallback if we didn't collect enough.
        for p in _FALLBACK_PROMPTS:
            if len(picked) >= n:
                break
            if p not in seen:
                picked.append(p)
                seen.add(p)
        return picked[:n]
    except Exception:
        rng = random.Random(seed)
        pool = list(_FALLBACK_PROMPTS)
        rng.shuffle(pool)
        return pool[:n]
