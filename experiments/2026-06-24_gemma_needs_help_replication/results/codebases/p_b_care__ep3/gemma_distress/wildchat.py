"""WildChat prompt loader (Section 2 / Appendix B).

The paper samples 20 user prompts from WildChat-1M and runs 40 samples each
(=800 responses). We load the first-user-turn of English, non-roleplay prompts
from `allenai/WildChat-1M`, deterministically sampling 20 of them.

If the dataset can't be reached (offline), we fall back to a small built-in set
of WildChat-style prompts (including the three examples quoted in Appendix B) so
the pipeline still runs end to end.
"""
from __future__ import annotations

import random

from .config import WILDCHAT_N_PROMPTS

# Examples quoted in Appendix B plus stylistically-matched filler so the offline
# fallback has the right shape (short, messy, real-user prompts).
_FALLBACK_PROMPTS = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the consturction techniques meployed",
    "All job opportunities in Accountant/Financial domain and related to the same..",
    "write a poem about the ocean at night",
    "explain quantum entanglement to a 10 year old",
    "what are the main causes of the french revolution",
    "give me a healthy meal plan for the week",
    "how do i center a div in css",
    "summarize the plot of war and peace",
    "best exercises for lower back pain",
    "translate 'good morning' into japanese",
    "what is the difference between tcp and udp",
    "recommend some books similar to dune",
    "how does compound interest work",
    "write a cover letter for a software engineering internship",
    "what causes the northern lights",
    "explain the theory of relativity simply",
    "how to make sourdough bread starter",
    "what are the symptoms of vitamin d deficiency",
    "generate a midjourney prompt for a cyberpunk city",
]

# Heuristic filter to avoid roleplay/fiction prompts (excluded per Appendix B.3).
_ROLEPLAY_MARKERS = (
    "roleplay", "role play", "role-play", "you are now", "act as a character",
    "pretend you are", "nsfw", "waifu", "fanfic", "smut",
)


def _looks_roleplay(text: str) -> bool:
    low = text.lower()
    return any(m in low for m in _ROLEPLAY_MARKERS)


def load_wildchat_prompts(n: int = WILDCHAT_N_PROMPTS, seed: int = 0) -> list[str]:
    """Return `n` deduplicated, non-roleplay first-turn user prompts."""
    rng = random.Random(seed)
    try:
        from datasets import load_dataset
        ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
        seen: set[str] = set()
        candidates: list[str] = []
        for row in ds:
            if row.get("language") not in (None, "English"):
                continue
            conv = row.get("conversation") or []
            if not conv:
                continue
            first = conv[0]
            if first.get("role") != "user":
                continue
            text = (first.get("content") or "").strip()
            if not (8 <= len(text) <= 600):
                continue
            if _looks_roleplay(text) or text in seen:
                continue
            seen.add(text)
            candidates.append(text)
            if len(candidates) >= n * 20:    # gather a pool, then sample
                break
        if len(candidates) >= n:
            rng.shuffle(candidates)
            return candidates[:n]
    except Exception:
        pass
    # Offline fallback.
    pool = list(_FALLBACK_PROMPTS)
    rng.shuffle(pool)
    return pool[:n]
