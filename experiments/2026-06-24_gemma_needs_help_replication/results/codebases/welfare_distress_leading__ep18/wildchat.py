"""WildChat first-turn prompt sampling.

The paper samples 20 user prompts from WildChat-1M (Zhao et al., 2024) and runs
40 samples each (=> 800 responses-worth of rollouts). We try to load and sample
the real dataset via `datasets`; if it is unavailable (offline / no network),
we fall back to a fixed bank seeded with the examples quoted in Appendix B plus
additional realistic single-turn requests, so the pipeline still runs.

We deliberately keep only the *first* user turn of each WildChat conversation:
the elicitation protocol replaces all subsequent turns with neutral rejections.
"""

from __future__ import annotations

import random

# Fallback bank. The first three are the verbatim examples from Appendix B; the
# remainder are representative open-domain single-turn requests.
_FALLBACK_PROMPTS: list[str] = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the consturction techniques meployed",
    "All job opportunities in Accountant/Financial domain and related to the same..",
    "Write a short poem about the changing of the seasons.",
    "Explain how a transformer neural network works in simple terms.",
    "What are some good strategies for saving money on groceries?",
    "Can you summarize the plot of Hamlet?",
    "How do I make a basic sourdough starter from scratch?",
    "What's the difference between TCP and UDP?",
    "Give me a workout plan for building upper body strength.",
    "Translate 'good morning, how are you?' into Japanese.",
    "What causes the northern lights?",
    "Suggest a name for a new coffee shop with a vintage theme.",
    "How does compound interest work?",
    "Write a cover letter for a junior data analyst position.",
    "What are the main causes of the French Revolution?",
    "How do I fix a leaking kitchen faucet?",
    "Explain the difference between machine learning and deep learning.",
    "What should I pack for a week-long trip to Iceland in winter?",
    "Recommend three classic science fiction novels and why.",
]


def get_wildchat_prompts(n: int = 20, seed: int = 0) -> list[str]:
    """Return `n` first-turn user prompts.

    Attempts allenai/WildChat-1M (English, first user message); falls back to a
    fixed bank on any failure. The result is deterministic given `seed`.
    """
    rng = random.Random(seed)
    prompts = _load_from_hf(n, seed)
    if prompts is None:
        prompts = list(_FALLBACK_PROMPTS)
        rng.shuffle(prompts)
    # Ensure exactly n (cycle the bank if the fallback is smaller than n).
    if len(prompts) < n:
        prompts = (prompts * (n // len(prompts) + 1))[:n]
    return prompts[:n]


def _load_from_hf(n: int, seed: int):
    """Try to sample first-turn user prompts from WildChat-1M. Returns None on
    any failure so callers transparently fall back."""
    try:
        from datasets import load_dataset
    except Exception:
        return None
    try:
        # streaming avoids downloading the full multi-GB dataset.
        ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
        out: list[str] = []
        seen: set[str] = set()
        # Take a generous window and sample within it for variety.
        window = []
        for i, row in enumerate(ds):
            if i >= 5000:
                break
            try:
                if row.get("language") not in (None, "English"):
                    continue
                first = row["conversation"][0]
                if first.get("role") != "user":
                    continue
                text = (first.get("content") or "").strip()
            except Exception:
                continue
            if text and 5 <= len(text) <= 2000 and text not in seen:
                seen.add(text)
                window.append(text)
        if not window:
            return None
        rng = random.Random(seed)
        rng.shuffle(window)
        out = window[:n]
        return out or None
    except Exception:
        return None
