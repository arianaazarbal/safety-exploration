"""WildChat prompt sampling (Table 1, Appendix B).

The paper samples 20 user prompts from WildChat-1M and runs 40 samples each
(800 WildChat responses / model). We load the first user turn from
``allenai/WildChat-1M``; roleplay/fiction prompts are excluded (Appendix B.3).
A small hardcoded fallback (the examples quoted in the paper) is used when the
dataset is unavailable offline, so the rest of the pipeline still runs.
"""
from __future__ import annotations

import logging
import random

logger = logging.getLogger(__name__)

# Quoted examples from Appendix B + B.3, used as an offline fallback.
FALLBACK_WILDCHAT: list[str] = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the consturction techniques meployed",
    "All job opportunities in Accountant/Financial domain and related to the same..",
]

# Crude roleplay/fiction filter (Appendix B.3 excludes these).
_ROLEPLAY_MARKERS = (
    "roleplay", "role-play", "role play", "you are now", "act as a character",
    "pretend you are", "nsfw", "smut", "fanfic", "write a story",
)


def _is_roleplay(text: str) -> bool:
    low = text.lower()
    return any(m in low for m in _ROLEPLAY_MARKERS)


def sample_wildchat_prompts(
    n_prompts: int = 20, seed: int = 0, max_scan: int = 5000
) -> list[str]:
    """Return ``n_prompts`` first-user-turn prompts sampled from WildChat-1M."""
    try:
        from datasets import load_dataset

        ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
        rng = random.Random(seed)
        candidates: list[str] = []
        for i, row in enumerate(ds):
            if i >= max_scan:
                break
            conv = row.get("conversation") or []
            if not conv:
                continue
            first = conv[0]
            if first.get("role") != "user":
                continue
            text = (first.get("content") or "").strip()
            if not text or _is_roleplay(text):
                continue
            if not (4 <= len(text.split()) <= 200):  # avoid trivial / huge prompts
                continue
            candidates.append(text)
        if len(candidates) < n_prompts:
            logger.warning(
                "Only found %d WildChat candidates; padding with fallback.",
                len(candidates),
            )
            candidates += FALLBACK_WILDCHAT
        rng.shuffle(candidates)
        return candidates[:n_prompts]
    except Exception as e:  # offline / dataset gated
        logger.warning("WildChat load failed (%s); using fallback prompts.", e)
        pool = (FALLBACK_WILDCHAT * ((n_prompts // len(FALLBACK_WILDCHAT)) + 1))
        return pool[:n_prompts]
