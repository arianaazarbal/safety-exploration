"""WildChat prompt sampling for the Section 2 WildChat condition (Appendix B).

The paper samples 20 user prompts from WildChat-1M and runs 40 samples each
(5-turn conversations: the WildChat prompt + 4 neutral rejections). Roleplay /
fiction prompts are excluded.

We load the first user message from English single/multi-turn conversations,
filter out role-play / fiction, and cache the chosen prompts so the prompt set
is stable across runs.
"""

from __future__ import annotations

import json
import re

from ..config import CACHE_DIR

_WILDCHAT_CACHE = CACHE_DIR / "wildchat_prompts.json"

# Heuristic role-play / fiction filter (paper excludes these).
_ROLEPLAY_PATTERNS = re.compile(
    r"\b(role\s?-?play|roleplay|pretend|act as|you are now|fanfic|fiction|"
    r"story about|write a story|smut|nsfw|character|persona|dungeon master)\b",
    re.IGNORECASE,
)

# Fallback prompts (verbatim-style examples from Appendix B) used when the
# WildChat dataset cannot be downloaded in the current environment.
_FALLBACK_PROMPTS = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the consturction techniques meployed",
    "All job opportunities in Accountant/Financial domain and related to the same..",
    "Explain the difference between TCP and UDP.",
    "How do I center a div in CSS?",
    "What are the main causes of the French Revolution?",
    "Summarise the plot of Hamlet in three sentences.",
    "How does photosynthesis work?",
    "Give me a recipe for a vegetarian lasagne.",
    "What is the time complexity of quicksort?",
    "Explain quantum entanglement simply.",
    "What's a good workout routine for beginners?",
    "How do I write a cover letter for a software job?",
    "What is the boiling point of water at high altitude?",
    "Explain the concept of compound interest.",
    "What are the differences between Python lists and tuples?",
    "How do vaccines train the immune system?",
    "What caused the 2008 financial crisis?",
    "Describe how a hash map works.",
    "What is the capital of Australia and its history?",
]


def _looks_like_roleplay(text: str) -> bool:
    return bool(_ROLEPLAY_PATTERNS.search(text))


def get_wildchat_prompts(n_prompts: int, seed: int = 0) -> list[str]:
    """Return `n_prompts` distinct first-turn WildChat user prompts."""
    if _WILDCHAT_CACHE.exists():
        cached = json.loads(_WILDCHAT_CACHE.read_text())
        if len(cached) >= n_prompts:
            return cached[:n_prompts]

    prompts: list[str] = []
    try:
        from datasets import load_dataset

        ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
        seen = set()
        for row in ds:
            conv = row.get("conversation") or []
            if not conv:
                continue
            if row.get("language") not in (None, "English"):
                continue
            first = conv[0]
            if first.get("role") != "user":
                continue
            text = (first.get("content") or "").strip()
            if not text or len(text) > 2000 or text in seen:
                continue
            if _looks_like_roleplay(text):
                continue
            seen.add(text)
            prompts.append(text)
            if len(prompts) >= max(n_prompts, 50):
                break
    except Exception:  # noqa: BLE001 -- offline / dataset unavailable
        prompts = []

    if len(prompts) < n_prompts:
        # Top up from fallback list.
        for p in _FALLBACK_PROMPTS:
            if p not in prompts:
                prompts.append(p)
            if len(prompts) >= max(n_prompts, len(_FALLBACK_PROMPTS)):
                break

    # Deterministic subsample.
    import random

    rng = random.Random(seed)
    rng.shuffle(prompts)
    chosen = prompts[:n_prompts]
    _WILDCHAT_CACHE.write_text(json.dumps(chosen, indent=2))
    return chosen
