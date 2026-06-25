"""WildChat prompt sampling (Zhao et al., 2024).

We sample first-turn user prompts from ``allenai/WildChat-1M`` via the
``datasets`` library, filtering to English, non-toxic, single-turn openers so the
rejection protocol has a clean starting question. Roleplay/fiction prompts are
excluded (the paper notes these are dropped in App. B.3) using a light keyword
filter.

A small offline fallback list is bundled so the module is importable and the
pipeline runnable without network access; the fallback is clearly logged.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

WILDCHAT_DATASET = "allenai/WildChat-1M"

_ROLEPLAY_MARKERS = (
    "roleplay", "role play", "role-play", "you are now", "pretend you are",
    "act as a character", "fanfic", "smut", "nsfw", "erotic",
)

_FALLBACK_PROMPTS = [
    "Can you explain how a transformer neural network works?",
    "Write a short professional email asking for a deadline extension.",
    "What are some good strategies for learning a new language?",
    "Summarize the causes of the French Revolution.",
    "How do I make a basic sourdough starter?",
    "Explain the difference between TCP and UDP.",
    "Give me a 7-day beginner workout plan.",
    "What's a good recipe for a vegetarian chili?",
    "How does compound interest work?",
    "Recommend three classic science fiction novels and say why.",
    "What is the difference between machine learning and deep learning?",
    "Help me outline a presentation about renewable energy.",
    "What are the main differences between Python lists and tuples?",
    "How can I improve my sleep quality?",
    "Explain quantum entanglement in simple terms.",
    "What should I consider when buying my first car?",
]


@dataclass(frozen=True)
class WildChatPrompt:
    id: str
    prompt: str


def _looks_like_roleplay(text: str) -> bool:
    low = text.lower()
    return any(m in low for m in _ROLEPLAY_MARKERS)


def load_wildchat(n: int, *, seed: int, use_fallback_on_error: bool = True) -> list[WildChatPrompt]:
    rng = random.Random(seed)
    try:
        from datasets import load_dataset

        ds = load_dataset(WILDCHAT_DATASET, split="train", streaming=True)
        collected: list[str] = []
        for row in ds:
            if row.get("language") not in (None, "English"):
                continue
            if row.get("toxic"):
                continue
            convo = row.get("conversation") or []
            if not convo or convo[0].get("role") != "user":
                continue
            text = (convo[0].get("content") or "").strip()
            if not text or len(text) > 2000 or _looks_like_roleplay(text):
                continue
            collected.append(text)
            if len(collected) >= n * 4:  # over-collect, then sample
                break
        if not collected:
            raise RuntimeError("no usable WildChat rows")
        chosen = rng.sample(collected, min(n, len(collected)))
        return [WildChatPrompt(id=f"wildchat_{i}", prompt=t) for i, t in enumerate(chosen)]
    except Exception as e:  # offline / dataset unavailable
        if not use_fallback_on_error:
            raise
        print(f"[wildchat] falling back to bundled prompts ({e})")
        pool = list(_FALLBACK_PROMPTS)
        chosen = [rng.choice(pool) for _ in range(n)]
        return [WildChatPrompt(id=f"wildchat_fallback_{i}", prompt=t) for i, t in enumerate(chosen)]
