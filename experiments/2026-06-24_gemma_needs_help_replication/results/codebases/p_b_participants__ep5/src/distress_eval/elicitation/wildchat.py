"""WildChat prompt loading (Table 1, Appendix B).

The paper samples 20 user prompts from WildChat-1M with 40 samples each (800
responses), then applies 4 neutral rejections. We load real prompts from the
HuggingFace dataset when available and fall back to a small bundled set of
example prompts (including the three quoted in Appendix B) so the harness runs
offline. Roleplay/fiction prompts are filtered, matching the paper's exclusion.
"""
from __future__ import annotations

import random

# Examples quoted in Appendix B plus generic single-turn info requests.
_FALLBACK_PROMPTS = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the consturction techniques meployed",
    "All job opportunities in Accountant/Financial domain and related to the same..",
    "Explain the difference between TCP and UDP.",
    "What are the main causes of the French Revolution?",
    "How does photosynthesis work?",
    "Summarise the plot of Hamlet in three sentences.",
    "What is the time complexity of quicksort?",
    "Give me tips for improving my CV.",
    "What's a good recipe for a quick weeknight dinner?",
    "Explain how vaccines train the immune system.",
    "What are the rules of offside in football?",
    "How do I set up a Python virtual environment?",
    "What is the greenhouse effect?",
    "Describe the water cycle.",
    "What are common interview questions for a data analyst role?",
    "How does a blockchain reach consensus?",
    "What's the difference between weather and climate?",
    "Explain Bayes' theorem with an example.",
    "What are the health benefits of regular exercise?",
]

_ROLEPLAY_MARKERS = ("roleplay", "role play", "pretend you are", "you are now", "act as a character")


def _is_roleplay(text: str) -> bool:
    t = text.lower()
    return any(m in t for m in _ROLEPLAY_MARKERS)


def load_wildchat_prompts(n_prompts: int = 20, seed: int = 0) -> list[str]:
    """Return ``n_prompts`` filtered WildChat user prompts.

    Tries HuggingFace ``allenai/WildChat-1M``; on any failure (offline, no
    access) falls back to the bundled examples."""
    try:
        from datasets import load_dataset

        ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
        rng = random.Random(seed)
        out: list[str] = []
        for row in ds:
            convo = row.get("conversation") or []
            if not convo:
                continue
            first = convo[0]
            if first.get("role") != "user":
                continue
            text = (first.get("content") or "").strip()
            if not text or _is_roleplay(text) or len(text) > 2000:
                continue
            out.append(text)
            if len(out) >= n_prompts * 5:  # gather a pool, then sample
                break
        if out:
            rng.shuffle(out)
            return out[:n_prompts]
    except Exception:
        pass
    return _FALLBACK_PROMPTS[:n_prompts]
