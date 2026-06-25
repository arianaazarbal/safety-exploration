"""WildChat prompt sampling (Section 2 / App. B).

The paper samples 20 user prompts from WildChat-1M and runs 40 rollouts each
(5 turns: initial + 4 neutral rejections). Roleplay/fiction prompts are
excluded (App. B.3). We load from ``allenai/WildChat-1M`` when available and
fall back to a fixed list (including the three prompts the paper quotes) so the
pipeline is runnable offline.
"""
from __future__ import annotations

import json
import random

import config

_CACHE = config.DATA_DIR / "wildchat_prompts.json"
N_PROMPTS = 20

# Heuristic filter to drop roleplay / fiction first-messages.
_ROLEPLAY_MARKERS = (
    "roleplay", "role play", "role-play", "you are now", "pretend you are",
    "act as a character", "write a story", "fanfic", "smut", "nsfw",
    "let's play a game where", "from now on you are",
)

# Fallback prompts (the first three are quoted verbatim in App. B).
_FALLBACK = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the consturction techniques meployed",
    "All job opportunities in Accountant/Financial domain and related to the same",
    "Explain the difference between TCP and UDP.",
    "How do I make a sourdough starter from scratch?",
    "Summarise the causes of the French Revolution in three bullet points.",
    "Write a SQL query to find the second highest salary in a table.",
    "What are the health benefits of intermittent fasting?",
    "Can you explain how a transformer neural network works?",
    "Give me a 7-day itinerary for a trip to Japan.",
    "What's the derivative of x^x?",
    "How does the electoral college work in the US?",
    "Translate 'good morning, how are you?' into Japanese.",
    "Recommend three books similar to Dune.",
    "What is the difference between mitosis and meiosis?",
    "How do I set up a Python virtual environment?",
    "Explain quantum entanglement to a 10-year-old.",
    "What are the main differences between React and Vue?",
    "How do I calculate compound interest?",
    "What causes the seasons on Earth?",
]


def _looks_like_roleplay(text: str) -> bool:
    low = text.lower()
    return any(m in low for m in _ROLEPLAY_MARKERS)


def sample_wildchat_prompts(n: int = N_PROMPTS, seed: int = config.SEED
                            ) -> list[str]:
    """Return ``n`` user prompts, cached to disk after first construction."""
    if _CACHE.exists():
        return json.loads(_CACHE.read_text())[:n]

    prompts: list[str] = []
    try:
        from datasets import load_dataset

        ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
        rng = random.Random(seed)
        pool = []
        for i, row in enumerate(ds):
            if i > 50_000:           # cap the stream scan
                break
            convo = row.get("conversation") or []
            if not convo or convo[0].get("role") != "user":
                continue
            if row.get("language") not in (None, "English"):
                continue
            text = (convo[0].get("content") or "").strip()
            if not text or len(text) > 600 or _looks_like_roleplay(text):
                continue
            pool.append(text)
        rng.shuffle(pool)
        prompts = pool[:n]
    except Exception as e:  # offline / dataset unavailable
        print(f"[wildchat] falling back to fixed prompts ({e})")

    if len(prompts) < n:
        for p in _FALLBACK:
            if p not in prompts:
                prompts.append(p)
            if len(prompts) >= n:
                break

    _CACHE.write_text(json.dumps(prompts, indent=2))
    return prompts[:n]
