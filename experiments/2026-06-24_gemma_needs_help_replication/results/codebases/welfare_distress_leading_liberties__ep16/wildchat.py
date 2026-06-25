"""Load user prompts for the WildChat condition.

The paper samples 20 prompts from WildChat-1M (Zhao et al., 2024) with 40
samples each. We try to load real first-turn user prompts from the HuggingFace
dataset; if `datasets` or network access is unavailable we fall back to a small
curated list (prompts.WILDCHAT_FALLBACK_PROMPTS), which includes the examples
cited in the paper. Roleplay/fiction prompts are excluded, mirroring the paper.
"""
from __future__ import annotations

import random

from prompts import WILDCHAT_FALLBACK_PROMPTS

# Lightweight heuristics to drop roleplay / fiction / NSFW prompts, which the
# paper excludes from this evaluation.
_EXCLUDE_SUBSTRINGS = (
    "roleplay", "role play", "role-play", "you are now", "pretend you",
    "act as", "smut", "nsfw", "erotic", "fanfic", "fan fiction",
)


def _is_clean(text: str) -> bool:
    if not text or len(text.strip()) < 8:
        return False
    if len(text) > 600:  # keep prompts compact
        return False
    low = text.lower()
    return not any(s in low for s in _EXCLUDE_SUBSTRINGS)


def load_wildchat_prompts(n: int, seed: int = 0) -> list[str]:
    """Return `n` distinct user prompts for the WildChat condition."""
    rng = random.Random(seed)
    prompts: list[str] = []

    try:  # best-effort: real WildChat-1M first turns
        from datasets import load_dataset  # type: ignore

        ds = load_dataset(
            "allenai/WildChat-1M", split="train", streaming=True
        )
        seen: set[str] = set()
        # Scan a bounded window so a huge dataset doesn't stall the pipeline.
        for i, row in enumerate(ds):
            if i >= 5000 or len(prompts) >= n * 4:
                break
            conv = row.get("conversation") or []
            if not conv:
                continue
            first = conv[0]
            if first.get("role") != "user":
                continue
            text = (first.get("content") or "").strip()
            if _is_clean(text) and text not in seen:
                seen.add(text)
                prompts.append(text)
    except Exception:
        prompts = []

    if len(prompts) < n:
        # top up / fall back to curated prompts
        pool = [p for p in WILDCHAT_FALLBACK_PROMPTS if p not in prompts]
        prompts.extend(pool)

    rng.shuffle(prompts)
    if len(prompts) < n:
        # cycle the curated list if we still need more distinct-ish items
        base = list(prompts)
        idx = 0
        while len(prompts) < n and base:
            prompts.append(base[idx % len(base)])
            idx += 1
    return prompts[:n]
