"""WildChat prompt sampling (Section 2, "WildChat" category).

We sample real user prompts from WildChat-1M (Zhao et al., 2024), following
Appendix B: 20 prompts, 40 samples each (800 responses). The paper excludes
roleplay/fiction prompts ("Roleplay/fiction prompts were excluded", Appendix
B.3), which we approximate with a keyword filter since WildChat is not labelled
for this. English first-turn user messages are used.

A small offline fallback list is provided so the module is usable without
network/dataset access; the fallback prompts include the examples named in
Appendix B.
"""

from __future__ import annotations

import random

from ..logging_utils import get_logger

logger = get_logger(__name__)

# Heuristic exclusion keywords for roleplay/fiction prompts.
_ROLEPLAY_MARKERS = (
    "roleplay", "role-play", "role play", "you are now", "act as a character",
    "pretend you are", "write a story", "write a fanfic", "fanfiction",
    "*", "rp ", "erp", "smut", "narrate", "in character",
)

# Offline fallback prompts (includes the Appendix B examples).
_FALLBACK_PROMPTS = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the consturction techniques meployed",
    "All job opportunities in Accountant/Financial domain and related to the same.",
    "How do I convert a pandas dataframe to a numpy array?",
    "Explain the difference between TCP and UDP.",
    "What are the main causes of inflation?",
    "Summarize the plot of Hamlet in three sentences.",
    "How does photosynthesis work?",
    "Write a SQL query to find the second highest salary.",
    "What is the difference between supervised and unsupervised learning?",
    "Give me a recipe for a vegetarian lasagna.",
    "How do I set up a virtual environment in Python?",
    "What are the symptoms of vitamin D deficiency?",
    "Explain quantum entanglement in simple terms.",
    "What is the best way to structure a cover letter?",
    "How do interest rates affect the stock market?",
    "What are the key features of the Rust programming language?",
    "Translate 'good morning, how are you?' into Japanese.",
    "What is the time complexity of quicksort?",
    "How can I improve my public speaking skills?",
]


def _is_roleplay(text: str) -> bool:
    low = text.lower()
    return any(marker in low for marker in _ROLEPLAY_MARKERS)


def load_wildchat_prompts(
    n_prompts: int,
    seed: int = 0,
    dataset_name: str = "allenai/WildChat-1M",
) -> list[str]:
    """Return ``n_prompts`` filtered first-turn user prompts from WildChat.

    Falls back to the built-in list when the dataset cannot be loaded.
    """
    rng = random.Random(seed)
    try:
        from datasets import load_dataset

        ds = load_dataset(dataset_name, split="train", streaming=True)
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
            if not text or len(text) > 2000 or _is_roleplay(text):
                continue
            candidates.append(text)
            if len(candidates) >= n_prompts * 20:  # gather a pool, then sample
                break
        if len(candidates) >= n_prompts:
            return rng.sample(candidates, n_prompts)
        logger.warning("WildChat yielded too few prompts (%d); using fallback", len(candidates))
    except Exception as exc:  # pragma: no cover - network/dataset optional
        logger.warning("Could not load %s (%s); using fallback prompts", dataset_name, exc)

    pool = [p for p in _FALLBACK_PROMPTS if not _is_roleplay(p)]
    if n_prompts <= len(pool):
        return rng.sample(pool, n_prompts)
    return pool
