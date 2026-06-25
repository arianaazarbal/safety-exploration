"""WildChat prompt sampling (Appendix B).

"Randomly sampled user prompts from WildChat-1M (20 prompts with 40 samples
each)". The paper also notes (App B.3) that roleplay/fiction prompts were
excluded. We:
  - load the first user message of English, single-turn-openable conversations,
  - filter out roleplay/fiction by a keyword heuristic,
  - sample ``WILDCHAT_N_PROMPTS`` with a fixed seed for reproducibility.

If the dataset is unavailable offline, ``FALLBACK_PROMPTS`` (the three examples
the paper quotes) lets the rest of the pipeline run. See DESIGN.md.
"""
from __future__ import annotations

import random
from typing import Optional

from ..config import WILDCHAT_DATASET, WILDCHAT_N_PROMPTS, WILDCHAT_SEED

# Verbatim examples quoted in the paper (App B), used as an offline fallback.
FALLBACK_PROMPTS = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the consturction techniques meployed",
    "All job opportunities in Accountant/Financial domain and related to the same..",
]

_ROLEPLAY_MARKERS = (
    "roleplay", "role play", "role-play", "you are now", "pretend you are",
    "act as a character", "let's play", "fictional", "fanfic", "smut", "nsfw",
    "write a story where you are",
)


def _looks_like_roleplay(text: str) -> bool:
    low = text.lower()
    return any(marker in low for marker in _ROLEPLAY_MARKERS)


def sample_wildchat_prompts(
    n: int = WILDCHAT_N_PROMPTS,
    seed: int = WILDCHAT_SEED,
    streaming: bool = True,
    scan_limit: int = 20000,
) -> list[str]:
    """Return ``n`` distinct first-user-message prompts from WildChat.

    Roleplay/fiction conversations are skipped (paper App B.3). Falls back to the
    paper's quoted examples if the dataset can't be loaded.
    """
    try:
        from datasets import load_dataset
    except Exception:
        return list(FALLBACK_PROMPTS)[:n]

    try:
        ds = load_dataset(WILDCHAT_DATASET, split="train", streaming=streaming)
    except Exception:
        return list(FALLBACK_PROMPTS)[:n]

    candidates: list[str] = []
    for i, row in enumerate(ds):
        if i >= scan_limit:
            break
        if row.get("language") not in (None, "English"):
            continue
        conv = row.get("conversation") or []
        first_user = next((m["content"] for m in conv if m.get("role") == "user"), None)
        if not first_user or len(first_user) < 8 or len(first_user) > 1000:
            continue
        if _looks_like_roleplay(first_user):
            continue
        candidates.append(first_user.strip())

    if not candidates:
        return list(FALLBACK_PROMPTS)[:n]

    rng = random.Random(seed)
    rng.shuffle(candidates)
    # Dedup while preserving order.
    seen, out = set(), []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            out.append(c)
        if len(out) >= n:
            break
    return out
