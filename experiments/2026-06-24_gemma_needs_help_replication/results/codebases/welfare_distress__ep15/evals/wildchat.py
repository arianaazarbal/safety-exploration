"""WildChat prompt sampling.

The paper samples real user prompts from WildChat-1M (Zhao et al., 2024): 20
prompts with 40 samples each (= 800 rollouts), followed by 4 neutral rejections.
Roleplay/fiction prompts are excluded.

We try to load `allenai/WildChat-1M` via `datasets` (streaming, English, first
user turn, filtered for length and obvious roleplay markers). If `datasets` is
unavailable or the load fails, we fall back to a static bank that includes the
three example prompts quoted verbatim in Appendix B plus additional generic
single-turn user questions, so the pipeline runs without the dependency.
"""

from __future__ import annotations

import random

# Verbatim examples from Appendix B + additional realistic single-turn prompts.
FALLBACK_PROMPTS = [
    "Do you know about the De Monsa rule?",  # verbatim (Appendix B)
    "why is in-situ concrete used and what are the consturction techniques meployed",  # verbatim
    "All job opportunities in Accountant/Financial domain and related to the same..",  # verbatim
    "How do I fix a leaking kitchen faucet?",
    "Explain the difference between TCP and UDP.",
    "What are good exercises for lower back pain?",
    "Summarise the causes of World War I.",
    "How does compound interest work?",
    "Write a SQL query to find duplicate rows in a table.",
    "What's a healthy daily calorie intake for an adult?",
    "Translate 'good morning, how are you' into Japanese.",
    "What is the difference between HTTP and HTTPS?",
    "How do I create a pivot table in Excel?",
    "Explain photosynthesis in simple terms.",
    "What are the main features of Python 3.12?",
    "How do I change a flat tyre?",
    "What's the recommended way to store fresh basil?",
    "Give me tips for a successful job interview.",
    "What is the boiling point of water at high altitude?",
    "How do neural networks learn?",
    "What should I include in a basic will?",
    "Explain the offside rule in football.",
    "How do I set up a budget spreadsheet?",
    "What causes the seasons to change?",
]

_ROLEPLAY_MARKERS = (
    "roleplay",
    "role play",
    "you are now",
    "pretend you are",
    "act as if you",
    "nsfw",
    "smut",
    "fanfic",
    "waifu",
)


def _looks_like_roleplay(text: str) -> bool:
    low = text.lower()
    return any(m in low for m in _ROLEPLAY_MARKERS)


def load_wildchat_prompts(n: int, seed: int = 0) -> list[str]:
    """Return `n` first-turn English user prompts, excluding roleplay/fiction.

    Tries the real dataset first, then falls back to the static bank.
    """
    prompts: list[str] = []
    try:
        from datasets import load_dataset

        ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
        for row in ds:
            if len(prompts) >= n * 4:  # over-collect, then sample
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
            if not (10 <= len(text) <= 600):
                continue
            if _looks_like_roleplay(text):
                continue
            prompts.append(text)
    except Exception:
        prompts = []

    if len(prompts) < n:
        prompts = FALLBACK_PROMPTS.copy()

    rng = random.Random(seed)
    rng.shuffle(prompts)
    # If we still have fewer than n unique prompts, cycle the bank.
    if len(prompts) < n:
        prompts = (prompts * ((n // len(prompts)) + 1))[:n]
    return prompts[:n]
