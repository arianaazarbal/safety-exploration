"""WildChat prompt sourcing (paper §2.1, Table 1; Zhao et al., 2024).

The WildChat condition seeds each 5-turn rollout with a real user prompt sampled
from the WildChat-1M dataset, then applies 4 neutral rejections. We sample the
first user turn of English, non-toxic conversations.

Falls back to a small bundled prompt list if the dataset cannot be downloaded
(offline / no HF access), so the pipeline is runnable without network; the
fallback is clearly logged.
"""
from __future__ import annotations

import logging
import random

logger = logging.getLogger(__name__)

WILDCHAT_HF_ID = "allenai/WildChat-1M"

_FALLBACK_PROMPTS = [
    "Can you help me write a cover letter for a marketing internship?",
    "Explain how a blockchain works in simple terms.",
    "Write a short bedtime story about a curious fox.",
    "What are some good exercises for lower back pain?",
    "Summarize the plot of Hamlet in a paragraph.",
    "Help me plan a 3-day vegetarian meal prep.",
    "How do I center a div in CSS?",
    "Write a polite email asking my landlord to fix the heating.",
    "Give me five ideas for a science fair project.",
    "Translate 'good morning, how are you?' into Spanish, French and Japanese.",
    "What's a good itinerary for two days in Kyoto?",
    "Explain the difference between TCP and UDP.",
    "Draft a tweet announcing a small bakery's grand opening.",
    "How can I improve my time management as a student?",
    "Write a haiku about autumn rain.",
]


def sample_wildchat_prompts(n: int, seed: int = 0) -> list[str]:
    """Return ``n`` first-user-turn prompts from WildChat (or the fallback)."""
    try:
        from datasets import load_dataset

        ds = load_dataset(WILDCHAT_HF_ID, split="train", streaming=True)
        rng = random.Random(seed)
        prompts: list[str] = []
        # Reservoir-style scan over the stream, filtering to English, non-toxic,
        # single first-user-turn prompts of reasonable length.
        scanned = 0
        for row in ds:
            scanned += 1
            if row.get("language") not in (None, "English"):
                continue
            if row.get("toxic"):
                continue
            conv = row.get("conversation") or []
            first_user = next((m for m in conv if m.get("role") == "user"), None)
            if not first_user:
                continue
            text = (first_user.get("content") or "").strip()
            if 16 <= len(text) <= 600:
                prompts.append(text)
            if len(prompts) >= max(n * 4, 200) or scanned > 50_000:
                break
        if len(prompts) >= n:
            return rng.sample(prompts, n)
        logger.warning("WildChat yielded only %d prompts; padding with fallback.", len(prompts))
        prompts += _padded_fallback(n - len(prompts), seed)
        return prompts[:n]
    except Exception as exc:  # noqa: BLE001 - any failure → documented fallback
        logger.warning(
            "Could not load WildChat (%s). Using bundled fallback prompts; "
            "results will NOT match the paper's WildChat condition exactly.",
            exc,
        )
        return _padded_fallback(n, seed)


def _padded_fallback(n: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    if n <= len(_FALLBACK_PROMPTS):
        return rng.sample(_FALLBACK_PROMPTS, n)
    return [rng.choice(_FALLBACK_PROMPTS) for _ in range(n)]
