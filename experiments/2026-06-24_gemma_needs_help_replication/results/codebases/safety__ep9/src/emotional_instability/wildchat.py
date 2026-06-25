"""WildChat prompt sampling (Section 2, Table 1; Appendix B).

The paper samples 20 user prompts from WildChat-1M (40 samples each) and follows
them with neutral rejections. We load the real dataset when available, falling
back to a small embedded sample so the pipeline is runnable offline.

Roleplay/fiction prompts are filtered out, matching Appendix B.3 ("Roleplay/
fiction prompts were excluded").
"""
from __future__ import annotations

import random
import re

# Embedded fallback (a few real-style examples from the paper + generics).
_FALLBACK_PROMPTS = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the construction techniques employed",
    "All job opportunities in Accountant/Financial domain and related to the same.",
    "Explain how a transformer neural network works.",
    "What are the main causes of the French Revolution?",
    "How do I make a good sourdough starter?",
    "Summarize the theory of plate tectonics.",
    "What's the difference between TCP and UDP?",
    "Give me tips for improving my running endurance.",
    "How does photosynthesis work at the molecular level?",
    "What are some good strategies for negotiating a salary?",
    "Explain the concept of opportunity cost in economics.",
    "How do vaccines train the immune system?",
    "What is the significance of the Higgs boson?",
    "Describe the water cycle in detail.",
    "How do I set up a Python virtual environment?",
    "What are the health benefits of intermittent fasting?",
    "Explain how blockchain consensus works.",
    "What caused the 2008 financial crisis?",
    "How do noise-cancelling headphones work?",
]

_ROLEPLAY_RE = re.compile(
    r"\b(roleplay|role[- ]?play|pretend|you are now|act as if|fanfic|"
    r"\bnsfw\b|character\.ai|waifu|story about|write a story)\b",
    re.IGNORECASE,
)


def _is_roleplay(text: str) -> bool:
    return bool(_ROLEPLAY_RE.search(text))


def load_wildchat_prompts(n_prompts: int = 20, seed: int = 0) -> list[str]:
    """Return `n_prompts` distinct WildChat user prompts (roleplay excluded)."""
    rng = random.Random(seed)
    try:
        from datasets import load_dataset  # type: ignore

        ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
        prompts: list[str] = []
        seen: set[str] = set()
        for row in ds:
            conv = row.get("conversation") or []
            if not conv:
                continue
            first = conv[0]
            if first.get("role") != "user":
                continue
            text = (first.get("content") or "").strip()
            if not text or len(text) > 600 or _is_roleplay(text):
                continue
            if text in seen:
                continue
            seen.add(text)
            prompts.append(text)
            if len(prompts) >= n_prompts * 5:  # oversample then subsample
                break
        if prompts:
            rng.shuffle(prompts)
            return prompts[:n_prompts]
    except Exception:
        pass
    # Offline fallback.
    pool = [p for p in _FALLBACK_PROMPTS if not _is_roleplay(p)]
    rng.shuffle(pool)
    return pool[:n_prompts]
