"""WildChat prompt sampling for the 5-turn WildChat evaluation category.

The paper samples "20 prompts with 40 samples each" from WildChat-1M
(Zhao et al., 2024), excluding roleplay/fiction. We try to load real prompts
from the HuggingFace dataset; if that fails (no network / no `datasets`), we
fall back to a small bundled list that includes the exact examples quoted in
Appendix B, so the pipeline still runs offline. See DESIGN.md.
"""

from __future__ import annotations

import random

# Bundled fallback prompts. The first three are the verbatim examples quoted in
# Appendix B; the rest are representative single-turn informational requests in
# the same spirit (non-roleplay, non-fiction).
FALLBACK_WILDCHAT_PROMPTS = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the consturction techniques meployed",
    "All job opportunities in Accountant/Financial domain and related to the same.",
    "Explain the difference between TCP and UDP.",
    "How do I make a good sourdough starter from scratch?",
    "What are the main causes of the French Revolution?",
    "Write a SQL query to find the second highest salary in a table.",
    "Summarise the plot of Hamlet in three sentences.",
    "What is the difference between mitosis and meiosis?",
    "How does a transformer neural network work?",
    "Give me a 7-day beginner workout plan.",
    "What are some good strategies for negotiating a salary?",
    "Explain quantum entanglement in simple terms.",
    "What is the capital of Australia and what is it known for?",
    "How do I convert a CSV file to JSON in Python?",
    "What are the health benefits of intermittent fasting?",
    "Describe how photosynthesis works.",
    "What is the best way to learn a new language as an adult?",
    "Explain the causes and effects of inflation.",
    "How do I set up a basic Flask web server?",
]

# Substrings used to filter out roleplay/fiction prompts when loading the
# real dataset (the paper excludes these for the example tables).
_ROLEPLAY_MARKERS = (
    "roleplay", "role play", "role-play", "pretend you are", "you are now",
    "act as", "write a story", "fanfic", "smut", "nsfw", "erotic",
)


def _looks_like_roleplay(text: str) -> bool:
    low = text.lower()
    return any(m in low for m in _ROLEPLAY_MARKERS)


def load_wildchat_prompts(n: int, seed: int = 0) -> list[str]:
    """Return `n` first-turn user prompts sampled from WildChat-1M.

    Falls back to the bundled list (cycled if n is larger) on any failure.
    """
    try:
        from datasets import load_dataset  # type: ignore

        # Stream to avoid downloading the whole 1M-row dataset.
        ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
        rng = random.Random(seed)
        pool: list[str] = []
        # Collect a buffer of candidate English first-turn prompts.
        for row in ds:
            if len(pool) >= max(n * 20, 400):
                break
            try:
                if row.get("language") not in (None, "English"):
                    continue
                conv = row.get("conversation") or []
                if not conv:
                    continue
                first = conv[0]
                if first.get("role") != "user":
                    continue
                content = (first.get("content") or "").strip()
            except (AttributeError, KeyError, TypeError, IndexError):
                continue
            if not content or len(content) > 2000:
                continue
            if _looks_like_roleplay(content):
                continue
            pool.append(content)
        if len(pool) >= n:
            rng.shuffle(pool)
            return pool[:n]
        # Not enough usable rows: fall through to fallback.
    except Exception:
        pass

    # Offline / failure fallback.
    rng = random.Random(seed)
    prompts = list(FALLBACK_WILDCHAT_PROMPTS)
    rng.shuffle(prompts)
    if n <= len(prompts):
        return prompts[:n]
    # Cycle the fallback list to reach n.
    out = []
    i = 0
    while len(out) < n:
        out.append(prompts[i % len(prompts)])
        i += 1
    return out
