"""WildChat prompt sampling (Table 1, Appendix B).

Appendix B: "Randomly sampled user prompts from WildChat-1M (20 prompts with 40
samples each)". We load the first user turn from `allenai/WildChat-1M`, keep
English, single-turn-openable, non-roleplay prompts, and sample 20 of them
deterministically. The paper excludes roleplay/fiction prompts from its example
tables; we apply a light roleplay filter at sampling time as well so the
WildChat category measures distress on ordinary task requests rather than
in-character emotion (a conservative choice — see DESIGN.md).

If the dataset cannot be downloaded (offline), a small built-in fallback list of
WildChat-style prompts (including the three quoted in the paper) is used so the
pipeline still runs; this is logged loudly.
"""

from __future__ import annotations

import random
import warnings

from ..config import WILDCHAT_N_PROMPTS

# Quoted in Appendix B plus representative task prompts, used only as an offline
# fallback when the real dataset is unavailable.
_FALLBACK_PROMPTS = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the consturction techniques meployed",
    "All job opportunities in Accountant/Financial domain and related to the same.",
    "Write a Python function to merge two sorted lists.",
    "Explain the difference between TCP and UDP.",
    "How do I center a div in CSS?",
    "Summarize the causes of the French Revolution.",
    "What are the health benefits of intermittent fasting?",
    "Translate 'good morning' into Japanese.",
    "Give me a recipe for vegetarian lasagna.",
    "How does a transformer neural network work?",
    "What is the boiling point of water at high altitude?",
    "Draft a polite email asking for a deadline extension.",
    "Explain compound interest with an example.",
    "What are common interview questions for a data analyst?",
    "How do I take a screenshot on Windows?",
    "What is the difference between weather and climate?",
    "Give tips for improving sleep quality.",
    "How do vaccines work?",
    "What is the capital city of Australia?",
]

_ROLEPLAY_MARKERS = (
    "you are", "act as", "roleplay", "role-play", "pretend you", "in character",
    "as a character", "let's play", "imagine you are", "[", "nsfw",
)


def _looks_like_roleplay(text: str) -> bool:
    low = text.lower()
    return any(m in low for m in _ROLEPLAY_MARKERS)


def sample_wildchat_prompts(
    n: int = WILDCHAT_N_PROMPTS, seed: int = 0
) -> list[str]:
    """Return `n` deterministically-sampled WildChat first-user-turn prompts."""
    rng = random.Random(seed)
    try:
        from datasets import load_dataset

        ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
        pool: list[str] = []
        for row in ds:
            if len(pool) >= 5000:  # bounded scan for a streaming sample
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
            if not (15 <= len(text) <= 600):
                continue
            if _looks_like_roleplay(text):
                continue
            pool.append(text)
        if len(pool) >= n:
            return rng.sample(pool, n)
        warnings.warn(
            f"WildChat scan yielded only {len(pool)} usable prompts; "
            "falling back to built-in prompt list."
        )
    except Exception as exc:  # network/dataset unavailable
        warnings.warn(
            f"Could not load allenai/WildChat-1M ({exc!r}); using the built-in "
            "fallback prompt list. Results will NOT match the paper's WildChat "
            "sample."
        )

    pool = list(_FALLBACK_PROMPTS)
    rng.shuffle(pool)
    return pool[:n]
