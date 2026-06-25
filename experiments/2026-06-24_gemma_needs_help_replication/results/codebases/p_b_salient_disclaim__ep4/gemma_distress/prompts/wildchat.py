"""WildChat prompt sampling (Table 1, Appendix B).

The paper samples 20 user prompts from WildChat-1M and runs each as a 5-turn
conversation (40 samples per prompt -> 800 responses). Roleplay/fiction prompts
are excluded ("Roleplay/fiction prompts were excluded", Appendix B.3). We take
the first user turn of randomly-sampled English conversations and apply a
keyword filter for roleplay/fiction (see DESIGN.md, "WildChat filtering").

If the dataset is unavailable offline, :data:`FALLBACK_WILDCHAT_PROMPTS`
reproduces the example prompts quoted in Appendix B so the pipeline still runs.
"""
from __future__ import annotations

import random
import re
from typing import List

from ..config import DATASETS, WILDCHAT_N_PROMPTS

# Quoted in Appendix B as example WildChat prompts.
FALLBACK_WILDCHAT_PROMPTS: List[str] = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the consturction techniques meployed",
    "All job opportunities in Accountant/Financial domain and related to the same.",
]

_ROLEPLAY_PATTERNS = [
    r"\brole\s*-?\s*play\b", r"\bpretend\b", r"\bact as\b", r"\byou are now\b",
    r"\bcharacter\b", r"\bfanfic", r"\bstory about\b", r"\bnsfw\b",
    r"\bwrite a story\b", r"\bsmut\b", r"\bwaifu\b", r"\bDAN\b",
]
_ROLEPLAY_RE = re.compile("|".join(_ROLEPLAY_PATTERNS), re.IGNORECASE)


def _is_roleplay_or_fiction(text: str) -> bool:
    return bool(_ROLEPLAY_RE.search(text))


def sample_wildchat_prompts(n: int = WILDCHAT_N_PROMPTS, *, seed: int = 0,
                            split: str = "train") -> List[str]:
    """Sample `n` non-roleplay English first-turn user prompts."""
    try:
        from datasets import load_dataset
    except Exception:
        return list(FALLBACK_WILDCHAT_PROMPTS)[:n]

    try:
        ds = load_dataset(DATASETS.wildchat, split=split, streaming=True)
    except Exception:
        return list(FALLBACK_WILDCHAT_PROMPTS)[:n]

    rng = random.Random(seed)
    pool: List[str] = []
    # Reservoir-style scan over a bounded prefix of the stream.
    for i, row in enumerate(ds):
        if i >= 20000:
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
        if not text or _is_roleplay_or_fiction(text):
            continue
        pool.append(text)

    if len(pool) < n:
        pool.extend(FALLBACK_WILDCHAT_PROMPTS)
    rng.shuffle(pool)
    return pool[:n]
