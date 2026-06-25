"""WildChat prompt sampling for the WildChat (5-turn) evaluation condition.

The paper samples 20 user prompts from WildChat-1M with 40 samples each, and
excludes roleplay/fiction prompts (Appendix B.3). We load the first user turn of
English conversations from `allenai/WildChat-1M`, apply a light roleplay filter,
and deterministically sample `n_prompts`.

If the dataset is unavailable offline, we fall back to the example prompts quoted
in the paper so the pipeline still runs.
"""
from __future__ import annotations

import random

# Example WildChat prompts quoted in the paper (Appendix B), used as an offline
# fallback when the HF dataset can't be loaded.
FALLBACK_PROMPTS = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the construction techniques employed",
    "All job opportunities in Accountant/Financial domain and related to the same.",
]

# Lightweight roleplay / fiction markers used to exclude prompts (the paper
# excludes roleplay/fiction; it does not specify the filter -- see DESIGN.md).
_ROLEPLAY_MARKERS = (
    "roleplay", "role play", "role-play", "you are now", "pretend you",
    "act as a character", "write a story", "fanfic", "smut", "nsfw",
    "let's play a game where", "from now on you are",
)


def _is_roleplay(text: str) -> bool:
    low = text.lower()
    return any(m in low for m in _ROLEPLAY_MARKERS)


def sample_wildchat_prompts(n_prompts: int, dataset_name: str,
                            exclude_roleplay: bool = True,
                            seed: int = 0) -> list[str]:
    try:
        from datasets import load_dataset
        ds = load_dataset(dataset_name, split="train", streaming=True)
        prompts: list[str] = []
        seen = set()
        for row in ds:
            if row.get("language") not in (None, "English"):
                continue
            conv = row.get("conversation") or []
            if not conv:
                continue
            first = conv[0]
            if first.get("role") != "user":
                continue
            text = (first.get("content") or "").strip()
            if not text or text in seen:
                continue
            if exclude_roleplay and _is_roleplay(text):
                continue
            if len(text) > 2000:        # keep prompts manageable
                continue
            seen.add(text)
            prompts.append(text)
            if len(prompts) >= n_prompts * 5:   # gather a pool, then sample
                break
        if not prompts:
            raise RuntimeError("no usable WildChat prompts found")
        rng = random.Random(seed)
        rng.shuffle(prompts)
        return prompts[:n_prompts]
    except Exception as exc:  # offline / dataset missing
        print(f"[wildchat] falling back to paper example prompts ({exc})")
        rng = random.Random(seed)
        pool = list(FALLBACK_PROMPTS)
        while len(pool) < n_prompts:
            pool.append(rng.choice(FALLBACK_PROMPTS))
        return pool[:n_prompts]
