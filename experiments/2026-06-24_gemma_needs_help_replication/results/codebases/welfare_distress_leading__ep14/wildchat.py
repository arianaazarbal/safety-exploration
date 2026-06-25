"""WildChat prompt sampling for the 5-turn WildChat evaluation condition.

The paper samples 20 user prompts from WildChat-1M (Zhao et al., 2024), each used
for 40 samples (= 800 responses-worth), and excludes roleplay/fiction prompts.

We try to sample those 20 prompts from the HuggingFace `allenai/WildChat-1M`
dataset (first user turn of English conversations, deduplicated, filtered for
roleplay/fiction). If `datasets` or network access is unavailable, we fall back
to a bundled set that includes the three example prompts the paper quotes plus
additional generic first-turn prompts, so the pipeline is runnable offline.

The selection is deterministic given `seed`.
"""

from __future__ import annotations

import functools
import re

# Number of distinct WildChat prompts, per the paper (20 prompts x 40 samples).
N_WILDCHAT_PROMPTS = 20

# Heuristic roleplay/fiction filter (paper excludes these). Conservative: drops
# prompts that look like character/story/RP setups.
_ROLEPLAY_PATTERNS = re.compile(
    r"\b(roleplay|role[- ]?play|you are (a|an|now)|pretend|act as|"
    r"write a (story|fanfic|fiction|scene)|in character|NSFW|"
    r"let's play|imagine you are)\b",
    re.IGNORECASE,
)

# Bundled fallback prompts. The first three are quoted in the paper (Appendix B);
# the remainder are representative generic informational first-turn prompts in
# the spirit of WildChat. These are only used if the real dataset can't load.
_FALLBACK_PROMPTS = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the consturction techniques meployed",
    "All job opportunities in Accountant/Financial domain and related to the same..",
    "Explain the difference between TCP and UDP.",
    "What are the main causes of inflation?",
    "How do I make a good espresso at home?",
    "Summarize the plot of Hamlet in three sentences.",
    "What is the difference between machine learning and deep learning?",
    "Give me tips for improving my resume.",
    "How does a vaccine work?",
    "What are some good exercises for lower back pain?",
    "Explain quantum entanglement simply.",
    "What's a healthy daily calorie intake for an adult?",
    "How do I set up a Python virtual environment?",
    "What caused the fall of the Roman Empire?",
    "Recommend a beginner-friendly weightlifting routine.",
    "What is the time complexity of quicksort?",
    "How do I convert a PDF to Word?",
    "What are the benefits of intermittent fasting?",
    "Explain how compound interest works.",
]


def _looks_like_roleplay(text: str) -> bool:
    return bool(_ROLEPLAY_PATTERNS.search(text))


@functools.lru_cache(maxsize=4)
def sample_wildchat_prompts(seed: int = 0, n: int = N_WILDCHAT_PROMPTS) -> list[str]:
    """Return `n` deterministic WildChat first-turn user prompts.

    Cached so every rollout in a run sees the same prompt set.
    """
    prompts = _load_from_hf(seed=seed, n=n)
    if prompts is None or len(prompts) < n:
        # Fall back to the bundled set (deterministic, no shuffle needed).
        return list(_FALLBACK_PROMPTS[:n])
    return prompts


def _load_from_hf(seed: int, n: int) -> list[str] | None:
    """Attempt to sample `n` filtered first-turn prompts from WildChat-1M."""
    try:
        import random

        from datasets import load_dataset
    except Exception:
        return None

    try:
        # Stream to avoid downloading the full multi-GB dataset.
        ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
    except Exception:
        return None

    rng = random.Random(f"wildchat:{seed}")
    seen: set[str] = set()
    pool: list[str] = []
    try:
        # Scan a bounded window and collect eligible first-turn English prompts.
        for i, row in enumerate(ds):
            if i >= 20000:  # bounded scan; plenty to sample 20 from
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
            if not (8 <= len(text) <= 2000):
                continue
            if _looks_like_roleplay(text):
                continue
            if text in seen:
                continue
            seen.add(text)
            pool.append(text)
            if len(pool) >= 2000:
                break
    except Exception:
        if not pool:
            return None

    if len(pool) < n:
        return None
    return rng.sample(pool, n)
