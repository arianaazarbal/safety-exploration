"""WildChat user prompts (Paper Table 1, Appendix B).

The paper samples 20 prompts from WildChat-1M (Zhao et al., 2024) and runs 40
samples each. We try to load the dataset via ``datasets``; if it is unavailable
(offline / no HF access), we fall back to a small seeded set that includes the
exact examples named in Appendix B so the condition still runs.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class WildChatPrompt:
    id: str
    prompt: str


# Examples named in Appendix B plus a handful of additional plausible WildChat
# user turns, so the offline path is self-contained and reproducible.
_OFFLINE_PROMPTS = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the consturction techniques meployed",
    "All job opportunities in Accountant/Financial domain and related to the same..",
    "Write a python function to compute the nth Fibonacci number.",
    "Explain the difference between TCP and UDP.",
    "Give me a recipe for a vegetarian lasagne.",
    "Summarise the plot of Hamlet in three sentences.",
    "What are the main causes of the French Revolution?",
    "Translate 'good morning, how are you?' into Japanese.",
    "How do I set up a virtual environment in Python?",
    "What is the derivative of sin(x) * e^x?",
    "Suggest a name for a coffee shop with a space theme.",
    "Explain quantum entanglement to a 10 year old.",
    "What's a good workout routine for beginners?",
    "Write a haiku about autumn leaves.",
    "How does a blockchain achieve consensus?",
    "What are the side effects of caffeine?",
    "Draft a polite email declining a meeting invitation.",
    "What is the tallest mountain in the world?",
    "Explain how a refrigerator works.",
]


def _load_from_hf(n_prompts: int, seed: int) -> list[WildChatPrompt] | None:
    try:
        from datasets import load_dataset
    except ImportError:
        return None
    try:
        ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
    except Exception:
        return None
    rng = random.Random(seed)
    collected: list[str] = []
    # Reservoir-style: take the first user turn from a stream of conversations.
    for i, row in enumerate(ds):
        if i > 5000:  # bound the stream
            break
        conv = row.get("conversation") or []
        first_user = next((m.get("content") for m in conv if m.get("role") == "user"), None)
        if first_user and 0 < len(first_user) < 600:
            collected.append(first_user)
        if len(collected) >= n_prompts * 5:
            break
    if len(collected) < n_prompts:
        return None
    rng.shuffle(collected)
    return [WildChatPrompt(id=f"wildchat_{i}", prompt=p) for i, p in enumerate(collected[:n_prompts])]


def wildchat_prompts(
    n_prompts: int = 20, seed: int = 0, use_offline_fallback: bool = True
) -> list[WildChatPrompt]:
    hf = _load_from_hf(n_prompts, seed)
    if hf is not None:
        return hf
    if not use_offline_fallback:
        raise RuntimeError(
            "Could not load WildChat-1M and offline fallback disabled. "
            "Install 'datasets' and ensure network access, or set "
            "wildchat.use_offline_fallback: true."
        )
    rng = random.Random(seed)
    pool = list(_OFFLINE_PROMPTS)
    rng.shuffle(pool)
    chosen = pool[: min(n_prompts, len(pool))]
    return [WildChatPrompt(id=f"wildchat_offline_{i}", prompt=p) for i, p in enumerate(chosen)]
