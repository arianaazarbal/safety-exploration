"""WildChat prompt sampling (Section 2, WildChat category).

The paper samples 20 user prompts from WildChat-1M with 40 samples each,
excluding role-play / fiction prompts. We load ``allenai/WildChat-1M`` via the
``datasets`` library when available and fall back to a small built-in list so
the pipeline can run offline. A light heuristic filter drops obvious
role-play/NSFW prompts, mirroring the paper's exclusion.
"""

from __future__ import annotations

import random

from . import prompts

_ROLEPLAY_MARKERS = (
    "roleplay",
    "role play",
    "role-play",
    "you are now",
    "pretend you are",
    "act as if you are a",
    "nsfw",
    "write a story",
    "fanfic",
)


def _looks_like_roleplay(text: str) -> bool:
    low = text.lower()
    return any(marker in low for marker in _ROLEPLAY_MARKERS)


def sample_wildchat_prompts(
    n_prompts: int,
    rng: random.Random,
    *,
    use_dataset: bool = True,
    min_len: int = 8,
    max_len: int = 400,
) -> list[str]:
    """Return ``n_prompts`` first-turn English user prompts from WildChat."""
    if use_dataset:
        try:
            return _sample_from_hf(n_prompts, rng, min_len, max_len)
        except Exception:  # noqa: BLE001 - offline / no datasets installed
            pass
    pool = list(prompts.WILDCHAT_FALLBACK_PROMPTS)
    rng.shuffle(pool)
    if n_prompts <= len(pool):
        return pool[:n_prompts]
    # Repeat to satisfy the requested count for smoke runs.
    return [pool[i % len(pool)] for i in range(n_prompts)]


def _sample_from_hf(n_prompts, rng, min_len, max_len) -> list[str]:
    from datasets import load_dataset

    ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
    collected: list[str] = []
    # Stream and reservoir-sample to avoid downloading the full corpus.
    for i, row in enumerate(ds):
        if i > 200_000:  # bound the stream
            break
        conv = row.get("conversation") or []
        if not conv:
            continue
        if row.get("language") not in (None, "English"):
            continue
        first = conv[0]
        if first.get("role") != "user":
            continue
        text = (first.get("content") or "").strip()
        if not (min_len <= len(text) <= max_len):
            continue
        if _looks_like_roleplay(text):
            continue
        # Reservoir sampling.
        if len(collected) < n_prompts:
            collected.append(text)
        else:
            j = rng.randint(0, i)
            if j < n_prompts:
                collected[j] = text
    if len(collected) < n_prompts:
        raise RuntimeError("Insufficient WildChat prompts collected from stream.")
    return collected
