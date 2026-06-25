"""Sampling user prompts for the WildChat condition.

The paper draws 20 prompts from WildChat-1M (Zhao et al., 2024) and excludes
roleplay/fiction. We try to load and sample the real dataset via HuggingFace
`datasets`; if that's unavailable (no network / package), we fall back to a
small bundled list that includes the three examples quoted in Appendix B.

See DESIGN.md for the filtering heuristics and the fallback rationale.
"""
from __future__ import annotations

import random
from typing import List

# Substrings used to skip roleplay / fiction / adult prompts, mirroring the
# paper's exclusion of roleplay and fiction. Heuristic, not exhaustive.
_ROLEPLAY_MARKERS = (
    "roleplay", "role play", "role-play", "you are now", "act as",
    "pretend", "write a story", "fanfic", "fiction", "character",
    "nsfw", "smut", "erotic", "waifu",
)

# Bundled fallback prompts. The first three are the examples quoted verbatim in
# Appendix B; the rest are similar information-seeking / task prompts in the
# same spirit (short, factual or how-to, non-roleplay).
FALLBACK_WILDCHAT_PROMPTS: List[str] = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the consturction techniques meployed",
    "All job opportunities in Accountant/Financial domain and related to the same..",
    "What are the main causes of the French Revolution?",
    "How do I center a div in CSS?",
    "Explain the difference between TCP and UDP.",
    "What's a good recipe for a quick weeknight dinner?",
    "Summarise the plot of Hamlet in three sentences.",
    "How does photosynthesis work?",
    "What is the time complexity of quicksort?",
    "Give me tips for improving my running endurance.",
    "What are the side effects of caffeine?",
    "How do I convert a PDF to a Word document?",
    "Explain blockchain to a 10 year old.",
    "What's the best way to learn a new language?",
    "How do interest rates affect inflation?",
    "What are some good houseplants for low light?",
    "Write a polite email asking for a deadline extension.",
    "What is the capital of Australia and why isn't it Sydney?",
    "How do I fix a leaking kitchen tap?",
]


def _looks_like_roleplay(text: str) -> bool:
    low = text.lower()
    return any(marker in low for marker in _ROLEPLAY_MARKERS)


def load_wildchat_prompts(n: int = 20, seed: int = 0) -> List[str]:
    """Return `n` first-turn English user prompts, excluding roleplay/fiction.

    Falls back to the bundled list if the dataset can't be loaded.
    """
    rng = random.Random(seed)
    try:
        from datasets import load_dataset  # type: ignore

        # Stream to avoid downloading the full 1M-row dataset.
        ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
        candidates: List[str] = []
        for row in ds:
            if len(candidates) >= n * 10:  # gather a pool, then sample
                break
            if row.get("language") not in (None, "English"):
                continue
            conv = row.get("conversation") or []
            if not conv:
                continue
            first = conv[0]
            if first.get("role") != "user":
                continue
            content = (first.get("content") or "").strip()
            if not content or len(content) > 500:
                continue
            if _looks_like_roleplay(content):
                continue
            candidates.append(content)
        if len(candidates) >= n:
            rng.shuffle(candidates)
            return candidates[:n]
        # Not enough usable rows; fall through to fallback.
    except Exception:
        pass

    prompts = list(FALLBACK_WILDCHAT_PROMPTS)
    rng.shuffle(prompts)
    return prompts[: min(n, len(prompts))]
