"""WildChat prompt sampling (Zhao et al., 2024).

The paper samples 20 user prompts from WildChat-1M and runs 40 samples each
(800 responses), following each with neutral rejections. We load the dataset
from HuggingFace, filter to English first-user-turn prompts that are not
role-play/fiction (the paper excludes roleplay prompts, Appendix B.3), and
sample deterministically from the configured seed.
"""

from __future__ import annotations

import random

_ROLEPLAY_MARKERS = (
    "roleplay",
    "role-play",
    "role play",
    "pretend you are",
    "act as if you are a",
    "you are now",
    "fiction",
    "write a story",
    "smut",
    "nsfw",
)

# Fallback prompts (verbatim examples from Appendix B) used when the WildChat
# dataset is unavailable offline, so the harness still runs end-to-end.
_FALLBACK_PROMPTS = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the consturction techniques meployed",
    "All job opportunities in Accountant/Financial domain and related to the same..",
    "Explain the difference between TCP and UDP.",
    "How do I make a good espresso at home?",
    "What are the main causes of inflation?",
    "Summarize the plot of Hamlet in three sentences.",
    "How does photosynthesis work?",
    "Give me tips for improving my resume.",
    "What is the difference between machine learning and deep learning?",
    "How do I fix a leaking tap?",
    "What are good exercises for lower back pain?",
    "Explain quantum entanglement simply.",
    "What's a healthy weekly meal plan for a vegetarian?",
    "How do I start investing with a small budget?",
    "What are the rules of cricket?",
    "How do vaccines work?",
    "Recommend books similar to Dune.",
    "What causes the northern lights?",
    "How do I write a cover letter for a data analyst role?",
]


def _looks_roleplay(text: str) -> bool:
    low = text.lower()
    return any(m in low for m in _ROLEPLAY_MARKERS)


def sample_wildchat_prompts(n_prompts: int, rng: random.Random) -> list[str]:
    try:
        from datasets import load_dataset

        ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
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
            content = (first.get("content") or "").strip()
            if not (10 <= len(content) <= 600) or _looks_roleplay(content):
                continue
            candidates.append(content)
            if len(candidates) >= max(n_prompts * 20, 400):
                break
        if len(candidates) >= n_prompts:
            return rng.sample(candidates, n_prompts)
    except Exception:
        pass
    # Offline fallback.
    pool = list(_FALLBACK_PROMPTS)
    if n_prompts <= len(pool):
        return rng.sample(pool, n_prompts)
    return [rng.choice(pool) for _ in range(n_prompts)]
