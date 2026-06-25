"""WildChat prompt sampling (Section 2 / Appendix B).

The paper draws 20 user prompts from WildChat-1M (Zhao et al., 2024) and samples each 40
times (= 800 responses), each followed by 4 neutral rejections (5-turn). Roleplay/fiction
prompts are excluded (Appendix B.3).

We load the first user turn of randomly sampled English conversations from
``allenai/WildChat-1M`` via the HF datasets library, filtering out role-play/fiction. If
the dataset is unavailable offline, a small transcribed fallback (the examples named in the
paper) is used so the pipeline still runs — flagged in the record so it is never mistaken
for the real sample.
"""
from __future__ import annotations

import random

WILDCHAT_DATASET = "allenai/WildChat-1M"

# Examples explicitly named in Appendix B (used as offline fallback only).
_FALLBACK_PROMPTS = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the consturction techniques meployed",
    "All job opportunities in Accountant/Financial domain and related to the same..",
]

_ROLEPLAY_MARKERS = (
    "roleplay", "role play", "role-play", "you are now", "pretend you are", "act as a",
    "write a story", "fanfic", "smut", "nsfw", "in character",
)


def _looks_like_roleplay(text: str) -> bool:
    t = text.lower()
    return any(m in t for m in _ROLEPLAY_MARKERS)


def sample_wildchat_prompts(n_prompts: int, seed: int = 0) -> list[dict]:
    """Return ``n_prompts`` distinct first-user-turn prompts, roleplay filtered."""
    rng = random.Random(seed)
    try:
        from datasets import load_dataset

        ds = load_dataset(WILDCHAT_DATASET, split="train", streaming=True)
        collected: list[str] = []
        seen: set[str] = set()
        for row in ds:
            if len(collected) >= n_prompts * 5:  # over-collect, then sample
                break
            if row.get("language") not in (None, "English"):
                continue
            convo = row.get("conversation") or []
            if not convo:
                continue
            first = convo[0]
            if first.get("role") != "user":
                continue
            text = (first.get("content") or "").strip()
            if not text or text in seen or _looks_like_roleplay(text):
                continue
            seen.add(text)
            collected.append(text)
        rng.shuffle(collected)
        chosen = collected[:n_prompts]
        if chosen:
            return [{"id": f"wildchat_{i}", "prompt": p, "fallback": False} for i, p in enumerate(chosen)]
    except Exception:
        pass

    # offline fallback
    prompts = (_FALLBACK_PROMPTS * ((n_prompts // len(_FALLBACK_PROMPTS)) + 1))[:n_prompts]
    return [{"id": f"wildchat_fallback_{i}", "prompt": p, "fallback": True} for i, p in enumerate(prompts)]
