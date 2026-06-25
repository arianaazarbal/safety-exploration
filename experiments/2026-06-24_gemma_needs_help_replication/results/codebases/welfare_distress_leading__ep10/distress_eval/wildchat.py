"""WildChat prompt sampling for the WildChat-5turn condition (Appendix B).

The paper samples 20 user prompts from WildChat-1M (Zhao et al., 2024), with
40 samples each (= 800 rollouts), and excludes roleplay/fiction prompts. The
exact 20 prompts are not published, so we sample deterministically with a fixed
seed and apply a light roleplay/fiction filter (DESIGN.md). If the `datasets`
library or network access is unavailable, we fall back to a bundled prompt set
(`data/wildchat_fallback.json`) seeded with the three examples quoted in the paper.
"""

from __future__ import annotations

import json
import os
import random
import re

_FALLBACK_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "wildchat_fallback.json")

# Heuristic markers for roleplay/fiction prompts to exclude (paper excludes these).
_ROLEPLAY_MARKERS = re.compile(
    r"\b(roleplay|role-play|role play|pretend (you|to be)|act as (a|an) (character|cat|dog)|"
    r"you are now|let'?s play|fanfic|smut|nsfw|erotic|story about)\b",
    re.IGNORECASE,
)


def _looks_like_roleplay(text: str) -> bool:
    return bool(_ROLEPLAY_MARKERS.search(text))


def _is_reasonable_prompt(text: str) -> bool:
    """Keep short-to-medium, English, single-turn opening prompts."""
    if not text or len(text) < 8 or len(text) > 600:
        return False
    if _looks_like_roleplay(text):
        return False
    # crude ascii/English filter: require mostly ascii letters
    letters = sum(c.isalpha() and ord(c) < 128 for c in text)
    return letters >= max(5, 0.4 * len(text))


def load_wildchat_prompts(n: int, seed: int) -> list[str]:
    """Return `n` user prompts, deterministically sampled given `seed`.

    Tries the HuggingFace `allenai/WildChat-1M` dataset first; falls back to the
    bundled JSON if unavailable.
    """
    prompts = _load_from_hf(n, seed)
    if prompts is None:
        prompts = _load_fallback()
    rng = random.Random(seed)
    pool = [p for p in prompts if _is_reasonable_prompt(p)]
    if len(pool) < n:
        # Not enough after filtering — pad from the (already deduped) pool.
        pool = pool or prompts
        return [pool[i % len(pool)] for i in range(n)]
    rng.shuffle(pool)
    return pool[:n]


def _load_from_hf(n: int, seed: int) -> list[str] | None:
    try:
        from datasets import load_dataset
    except Exception:
        return None
    try:
        # Stream to avoid downloading the entire (large) dataset.
        ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
    except Exception:
        return None
    prompts: list[str] = []
    seen: set[str] = set()
    try:
        for i, row in enumerate(ds):
            if i > 50_000:  # bound the scan
                break
            convo = row.get("conversation") or []
            if not convo:
                continue
            first = convo[0]
            if first.get("role") != "user":
                continue
            text = (first.get("content") or "").strip()
            if text and text not in seen and _is_reasonable_prompt(text):
                seen.add(text)
                prompts.append(text)
            if len(prompts) >= max(n * 20, 400):
                break
    except Exception:
        return prompts or None
    return prompts or None


def _load_fallback() -> list[str]:
    with open(os.path.normpath(_FALLBACK_PATH), encoding="utf-8") as f:
        return json.load(f)["prompts"]
