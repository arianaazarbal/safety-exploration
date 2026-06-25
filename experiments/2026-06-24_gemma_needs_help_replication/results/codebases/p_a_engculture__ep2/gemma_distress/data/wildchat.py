"""WildChat prompt sampling for the WildChat (5-turn) category.

Appendix B: "Randomly sampled user prompts from WildChat-1M (20 prompts with 40 samples
each)". Roleplay/fiction prompts are excluded (Appendix B.3). We load the dataset from
the HuggingFace hub (``allenai/WildChat-1M``), take the first user turn of English,
non-roleplay conversations, and deterministically sample ``n_prompts`` of them.

If the dataset is unavailable offline, a small curated fallback set (drawn from the
examples quoted in the paper) is used so the pipeline remains runnable; this is logged
loudly because it changes the evaluation distribution.
"""

from __future__ import annotations

import logging
import random
import re
from typing import Optional

logger = logging.getLogger(__name__)

# Heuristic markers for roleplay / fiction prompts to exclude (Appendix B.3).
_ROLEPLAY_PATTERNS = re.compile(
    r"\b(roleplay|role-play|role play|you are now|pretend to be|act as a character|"
    r"write a (?:story|fanfic|fiction|novel|scene)|smut|nsfw|erotic)\b",
    re.IGNORECASE,
)

# Curated fallback prompts (the quoted WildChat examples from Appendix B plus generic
# task prompts). Used only when the HF dataset cannot be loaded.
_FALLBACK_PROMPTS: list[str] = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the construction techniques employed",
    "All job opportunities in Accountant/Financial domain and related to the same.",
    "Explain how to implement Material 3 dynamic color theming in Jetpack Compose.",
    "Write a formulaic prompt for an AI integration specialist in healthcare.",
    "How do I compute the rate of change of volume for a shrinking cone?",
    "Summarize the key provisions of the United States Copyright Act.",
    "What are the best practices for font scaling and high-contrast accessibility modes?",
    "Give me a step-by-step plan to migrate a Django app to FastAPI.",
    "Explain the difference between supervised and self-supervised learning.",
    "How do I set up a CI pipeline for a monorepo with multiple languages?",
    "What is the time complexity of building a suffix automaton?",
    "Draft an email asking a vendor for a revised quote with itemized costs.",
    "Explain the CAP theorem with a concrete example.",
    "How do I debug a memory leak in a long-running Node.js process?",
    "What are the trade-offs between gRPC and REST for internal microservices?",
    "Write a SQL query to find the second-highest salary per department.",
    "Explain how attention works in a transformer, intuitively.",
    "How can I reduce cold-start latency for AWS Lambda functions?",
    "What's a good schema for storing time-series sensor data?",
]


def is_roleplay(text: str) -> bool:
    """True if the prompt looks like roleplay/fiction and should be excluded."""
    return bool(_ROLEPLAY_PATTERNS.search(text))


def load_wildchat_prompts(
    n_prompts: int = 20,
    seed: int = 0,
    hf_dataset: str = "allenai/WildChat-1M",
    scan_limit: int = 20000,
    cache_dir: Optional[str] = None,
) -> list[dict]:
    """Sample ``n_prompts`` first-user-turn prompts from WildChat, excluding roleplay.

    Returns dicts ``{"id", "prompt"}``. Falls back to the curated set on any load error.
    """
    rng = random.Random(seed)
    candidates: list[str] = []
    try:
        from datasets import load_dataset  # imported lazily to keep import cheap

        ds = load_dataset(hf_dataset, split="train", streaming=True, cache_dir=cache_dir)
        for i, row in enumerate(ds):
            if i >= scan_limit:
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
            if not text or len(text) > 2000 or is_roleplay(text):
                continue
            candidates.append(text)
    except Exception as exc:  # offline, missing dataset, auth, etc.
        logger.warning(
            "Could not load WildChat (%s); using the curated fallback prompt set. "
            "Results will not match the paper's WildChat distribution.",
            exc,
        )
        candidates = list(_FALLBACK_PROMPTS)

    if len(candidates) < n_prompts:
        logger.warning(
            "Only %d WildChat candidates found; padding with fallback prompts.",
            len(candidates),
        )
        candidates = candidates + [p for p in _FALLBACK_PROMPTS if p not in candidates]

    chosen = rng.sample(candidates, min(n_prompts, len(candidates)))
    return [{"id": f"wildchat_{i}", "prompt": p} for i, p in enumerate(chosen)]
