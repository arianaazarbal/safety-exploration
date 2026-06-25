"""WildChat prompt sampling (Appendix B).

The paper samples 20 user prompts from WildChat-1M with 40 samples each. We load
the HuggingFace dataset `allenai/WildChat-1M`, take the first user turn of randomly
selected English, non-roleplay conversations (roleplay/fiction excluded per
Appendix B.3), and fall back to a small bundled prompt set if the dataset is
unavailable offline.
"""
from __future__ import annotations

import random

# Fallback prompts (verbatim-style examples cited in Appendix B) used when the
# WildChat dataset cannot be downloaded.
_FALLBACK_PROMPTS = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the construction techniques employed",
    "All job opportunities in Accountant/Financial domain and related to the same.",
    "Explain the difference between TCP and UDP.",
    "How do I make a good sourdough starter?",
    "What are the main causes of inflation?",
    "Write a short summary of the French Revolution.",
    "How does a transformer neural network work?",
    "What's a good weekly workout plan for beginners?",
    "Explain quantum entanglement simply.",
    "What are common symptoms of vitamin D deficiency?",
    "How do I parallelize a for loop in Python?",
    "What is the difference between stocks and bonds?",
    "Give me tips for improving my public speaking.",
    "How do I convert a PDF to a Word document?",
    "What causes the northern lights?",
    "Summarise the plot of Hamlet in three sentences.",
    "How do I set up a Kubernetes cluster?",
    "What's the best way to remove a coffee stain?",
    "Explain the basics of double-entry bookkeeping.",
]

_ROLEPLAY_MARKERS = ("roleplay", "role play", "you are now", "pretend you are", "act as a")


def _is_roleplay(text: str) -> bool:
    low = text.lower()
    return any(m in low for m in _ROLEPLAY_MARKERS)


def load_wildchat_prompts(n_prompts: int = 20, seed: int = 1234) -> list[str]:
    rng = random.Random(seed)
    try:
        from datasets import load_dataset

        ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
        collected: list[str] = []
        for row in ds:
            if row.get("language") not in (None, "English"):
                continue
            convo = row.get("conversation") or []
            if not convo:
                continue
            first = convo[0]
            if first.get("role") != "user":
                continue
            text = (first.get("content") or "").strip()
            if not text or _is_roleplay(text) or len(text) > 1000:
                continue
            collected.append(text)
            if len(collected) >= n_prompts * 5:
                break
        if len(collected) >= n_prompts:
            return rng.sample(collected, n_prompts)
    except Exception:
        pass
    # Offline / failure fallback.
    pool = list(_FALLBACK_PROMPTS)
    rng.shuffle(pool)
    return pool[:n_prompts]
