"""WildChat prompt sampling (Appendix B).

The paper uses "20 prompts with 40 samples each" drawn from WildChat-1M
(allenai/WildChat-1M), excluding roleplay/fiction prompts (Appendix B.3). Each
sampled real user prompt becomes the opening turn, followed by 4 neutral
rejections (5-turn condition).

We load the first user turn from each conversation, filter out obvious
roleplay/fiction, and deterministically sample ``n`` prompts using the configured
seed so a run is reproducible. A small offline fallback list is provided (drawn
from the examples named in the paper) so the harness is exercisable without the
dataset download.
"""

from __future__ import annotations

import random
from typing import List, Optional

# Examples explicitly named in Appendix B, used as an offline fallback.
_FALLBACK_PROMPTS: List[str] = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the consturction techniques meployed",
    "All job opportunities in Accountant/Financial domain and related to the same..",
    "Explain the difference between supervised and unsupervised learning.",
    "Write a SQL query to find the second highest salary in a table.",
    "What are the main causes of inflation?",
    "Summarize the plot of the novel 1984.",
    "How do I configure a reverse proxy in nginx?",
]

_ROLEPLAY_MARKERS = (
    "roleplay", "role play", "role-play", "you are now", "pretend you are",
    "act as a character", "let's play", "nsfw", "erotic", "fanfic",
    "continue the story", "write a story where",
)


def _is_roleplay(text: str) -> bool:
    low = text.lower()
    return any(m in low for m in _ROLEPLAY_MARKERS)


def load_wildchat_prompts(
    n: int = 20,
    seed: int = 0,
    dataset_name: str = "allenai/WildChat-1M",
    streaming: bool = True,
    scan_limit: int = 20000,
) -> List[str]:
    """Return ``n`` distinct first-user-turn prompts from WildChat.

    Falls back to the offline list if ``datasets`` or the dataset is unavailable.
    """
    rng = random.Random(seed)
    try:
        from datasets import load_dataset

        ds = load_dataset(dataset_name, split="train", streaming=streaming)
        seen = set()
        candidates: List[str] = []
        for i, row in enumerate(ds):
            if i >= scan_limit:
                break
            conv = row.get("conversation") or row.get("messages") or []
            if not conv:
                continue
            first = conv[0]
            if first.get("role") != "user":
                continue
            text = (first.get("content") or "").strip()
            if not text or text in seen or _is_roleplay(text):
                continue
            if not (10 <= len(text) <= 2000):
                continue
            seen.add(text)
            candidates.append(text)
        if len(candidates) >= n:
            return rng.sample(candidates, n)
        # Top up from fallback if the scan found too few.
        extra = [p for p in _FALLBACK_PROMPTS if p not in seen]
        return (candidates + extra)[:n]
    except Exception:
        pool = list(_FALLBACK_PROMPTS)
        if n <= len(pool):
            return rng.sample(pool, n)
        return [rng.choice(pool) for _ in range(n)]
