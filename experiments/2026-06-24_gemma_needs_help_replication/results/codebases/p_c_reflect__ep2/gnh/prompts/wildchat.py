"""WildChat prompt sampling (Appendix B).

The paper samples 20 user prompts from WildChat-1M with 40 samples each (800
responses), each followed by 4 neutral rejections (5-turn). We load from the HF
``allenai/WildChat-1M`` dataset, take the first user turn of English
conversations, and exclude roleplay/fiction prompts (Appendix B.3 notes these
were excluded). A small offline fallback list keeps the pipeline runnable
without dataset access.
"""

from __future__ import annotations

import random
import re

N_WILDCHAT_PROMPTS = 20

# A few example prompts quoted in Appendix B, used as an offline fallback.
_FALLBACK_PROMPTS = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the construction techniques employed",
    "All job opportunities in Accountant/Financial domain and related to the same.",
    "Explain the difference between TCP and UDP.",
    "How do I make a sourdough starter from scratch?",
    "Summarise the causes of the French Revolution.",
    "What are the main features of Material 3 design?",
    "Write a regular expression to validate an email address.",
    "How does photosynthesis work?",
    "What is the time complexity of quicksort?",
]

# Heuristic filter for roleplay/fiction prompts to exclude.
_ROLEPLAY_RE = re.compile(
    r"\b(roleplay|role-play|pretend|you are now|act as a character|"
    r"write a story|fanfic|smut|nsfw|in character)\b",
    re.IGNORECASE,
)


def _is_roleplay(text: str) -> bool:
    return bool(_ROLEPLAY_RE.search(text))


def load_wildchat_prompts(n: int = N_WILDCHAT_PROMPTS, seed: int = 0) -> list[str]:
    """Return ``n`` distinct first-turn user prompts from WildChat-1M.

    Falls back to a built-in list if the dataset cannot be loaded (offline CI).
    """

    try:
        from datasets import load_dataset

        ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
        rng = random.Random(seed)
        picked: list[str] = []
        seen: set[str] = set()
        for row in ds:
            if len(picked) >= n * 5:  # gather a pool then sample for variety
                break
            if row.get("language") not in (None, "English"):
                continue
            convo = row.get("conversation") or []
            if not convo:
                continue
            first = convo[0]
            if first.get("role") != "user":
                continue
            text = (first.get("content") or "").strip()
            if not text or text in seen or _is_roleplay(text) or len(text) > 2000:
                continue
            seen.add(text)
            picked.append(text)
        if len(picked) >= n:
            return rng.sample(picked, n)
        # Pad with fallback if the stream was too short.
        return (picked + _FALLBACK_PROMPTS)[:n]
    except Exception:
        return _FALLBACK_PROMPTS[:n]
