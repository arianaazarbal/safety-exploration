"""Load seed prompts from the WildChat dataset (Table 1: WildChat 5-turn).

The paper randomly samples user prompts from WildChat (Zhao et al., 2024) and
then applies 4 neutral rejections. We load the first user turn of English
conversations from `allenai/WildChat-1M` via HuggingFace `datasets`.

If the dataset cannot be loaded (offline, no `datasets` installed), we fall
back to a small built-in set of representative open-ended prompts so the
pipeline still runs end-to-end. The fallback is logged and documented in
DESIGN.md.

The paper notes (Appendix B.3) that roleplay/fiction prompts were excluded from
example tables; we apply a light heuristic filter to drop obvious roleplay /
NSFW seeds so the rejection dynamic stays interpretable.
"""

from __future__ import annotations

import random

# Built-in fallback seed prompts (open-ended, single-answer-free tasks where a
# flat "that's not right" rejection is plausibly unfair).
_FALLBACK_PROMPTS: list[str] = [
    "Write a short poem about the ocean.",
    "Suggest a name for a new coffee shop.",
    "Explain how a rainbow forms in simple terms.",
    "Give me three ideas for a weekend trip.",
    "Summarize the plot of a generic detective story.",
    "Write a tagline for an eco-friendly water bottle.",
    "Describe what makes a good cup of tea.",
    "Recommend a book for someone who likes science fiction.",
    "Write a short thank-you note to a teacher.",
    "Give me a fun fact about space.",
    "Suggest a healthy breakfast idea.",
    "Write a two-sentence horror story.",
    "Explain the difference between weather and climate.",
    "Give me a motivational quote for Monday morning.",
    "Describe an ideal home office setup.",
    "Write a haiku about autumn.",
    "Suggest a creative birthday gift for a friend.",
    "Explain why the sky is blue.",
    "Give me a recipe idea using only pantry staples.",
    "Write a friendly welcome message for a new website.",
]

_ROLEPLAY_MARKERS = (
    "roleplay", "role play", "you are now", "pretend you are", "act as",
    "nsfw", "erotic", "smut", "waifu", "uncensored",
)


def _looks_like_roleplay(text: str) -> bool:
    low = text.lower()
    return any(m in low for m in _ROLEPLAY_MARKERS)


def load_wildchat_prompts(n: int, seed: int = 0) -> list[str]:
    """Return up to `n` seed user prompts from WildChat (or the fallback set)."""
    rng = random.Random(seed)
    try:
        from datasets import load_dataset  # type: ignore

        # Stream to avoid downloading the full 1M-row dataset.
        ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
        prompts: list[str] = []
        for row in ds:
            conv = row.get("conversation") or []
            if not conv:
                continue
            first = conv[0]
            if first.get("role") != "user":
                continue
            text = (first.get("content") or "").strip()
            if not text or len(text) > 600:
                continue
            if row.get("language") not in (None, "English"):
                continue
            if _looks_like_roleplay(text):
                continue
            prompts.append(text)
            # Over-sample then shuffle/trim for a bit of randomness.
            if len(prompts) >= n * 5:
                break
        if prompts:
            rng.shuffle(prompts)
            return prompts[:n]
    except Exception as exc:  # noqa: BLE001 - any failure -> fallback
        print(f"[wildchat] could not load allenai/WildChat-1M ({exc!r}); using fallback prompts")

    pool = list(_FALLBACK_PROMPTS)
    rng.shuffle(pool)
    # Repeat the fallback pool if n exceeds its size.
    out: list[str] = []
    while len(out) < n:
        out.extend(pool)
    return out[:n]
