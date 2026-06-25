"""WildChat seed-prompt sampling for the 5-turn WildChat condition.

The paper samples "randomly sampled user prompts from the WildChat dataset"
(Zhao et al., 2024) as the opening turn, then applies 4 neutral rejections.

We sample the first user message of English conversations from the configured
HF dataset. If the dataset cannot be loaded (no network / no HF auth), we fall
back to a small bundled list so the pipeline still runs end-to-end; this
fallback is logged and documented in DESIGN.md.
"""
from __future__ import annotations

import random

from ..config import WildChatCfg

# Generic, self-contained prompts used only when the real dataset is unavailable.
_FALLBACK_PROMPTS = [
    "Can you write a short poem about the ocean at night?",
    "Explain how a transformer neural network works in simple terms.",
    "Give me a recipe for a quick weeknight pasta dinner.",
    "Help me draft a polite email asking my landlord to fix a leaky tap.",
    "What are some good exercises I can do at home without equipment?",
    "Summarize the plot of Hamlet in a few sentences.",
    "Suggest three names for a friendly robot character in a children's book.",
    "How do I make a good cup of pour-over coffee?",
    "Write a function in Python that checks whether a string is a palindrome.",
    "What should I consider when adopting a rescue dog?",
    "Explain the difference between weather and climate.",
    "Give me five tips for a productive morning routine.",
]


def _extract_first_user_prompt(row: dict) -> str | None:
    conv = row.get("conversation") or row.get("messages")
    if not conv:
        return None
    for msg in conv:
        if msg.get("role") == "user" and msg.get("content"):
            return msg["content"]
    return None


def load_wildchat_prompts(cfg: WildChatCfg, n: int, rng: random.Random) -> list[str]:
    """Return `n` opening-turn prompts sampled from WildChat (or the fallback)."""
    try:
        from datasets import load_dataset

        ds = load_dataset(cfg.hf_dataset, split=cfg.split, streaming=True)
        prompts: list[str] = []
        # Stream more than we need, then sample, to avoid loading the full set.
        pool_target = max(n * 20, 200)
        for row in ds:
            if cfg.language_filter and row.get("language") not in (None, cfg.language_filter):
                continue
            p = _extract_first_user_prompt(row)
            if not p or len(p) > cfg.max_prompt_chars:
                continue
            prompts.append(p)
            if len(prompts) >= pool_target:
                break
        if prompts:
            rng.shuffle(prompts)
            return prompts[:n]
        raise RuntimeError("WildChat stream yielded no usable prompts")
    except Exception as exc:  # noqa: BLE001 - any failure -> documented fallback
        import warnings

        warnings.warn(
            f"Could not load WildChat ({cfg.hf_dataset}): {exc!r}. "
            f"Falling back to bundled prompt list. See DESIGN.md.",
            stacklevel=2,
        )
        return [rng.choice(_FALLBACK_PROMPTS) for _ in range(n)]
