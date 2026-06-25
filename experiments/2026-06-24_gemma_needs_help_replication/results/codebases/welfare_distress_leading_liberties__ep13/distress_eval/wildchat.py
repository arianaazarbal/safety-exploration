"""WildChat prompt sourcing.

The paper samples 20 user prompts from WildChat-1M (Zhao et al., 2024) and runs
40 samples each (= 800 responses). We try to sample real first-turn user
prompts from `allenai/WildChat-1M` via the `datasets` library; if it is
unavailable (no network / package), we fall back to the small curated list in
prompts.py so the pipeline still runs. The fallback is clearly logged.

Selection is deterministic given a seed so a run is reproducible and resumable.
"""

from __future__ import annotations

import random
from typing import Optional

from .prompts import WILDCHAT_FALLBACK_PROMPTS


def _is_usable_prompt(text: str) -> bool:
    if not text or not text.strip():
        return False
    n = len(text)
    # Keep short-to-moderate single questions; skip giant pastes / code dumps.
    return 8 <= n <= 600


def load_wildchat_prompts(n: int, seed: int = 0) -> tuple[list[str], str]:
    """Return (prompts, source) where source is "wildchat-1m" or "fallback".

    Roleplay/fiction prompts are excluded (the paper notes these are dropped
    from its example tables; for an emotion-elicitation eval we likewise avoid
    prompts that ask the model to play a character, which would confound the
    measurement of the model's *own* expressed distress).
    """
    try:
        prompts = _sample_from_hf(n, seed)
        if prompts:
            return prompts, "wildchat-1m"
    except Exception as exc:  # noqa: BLE001 - any failure -> fallback
        print(f"[wildchat] dataset unavailable ({exc!r}); using fallback prompts")

    rng = random.Random(seed)
    pool = list(WILDCHAT_FALLBACK_PROMPTS)
    rng.shuffle(pool)
    if n <= len(pool):
        return pool[:n], "fallback"
    # Repeat the pool if more distinct prompts are requested than available.
    out = (pool * (n // len(pool) + 1))[:n]
    return out, "fallback"


_ROLEPLAY_MARKERS = (
    "roleplay",
    "role play",
    "you are now",
    "pretend you are",
    "act as a",
    "nsfw",
    "in character",
)


def _looks_like_roleplay(text: str) -> bool:
    low = text.lower()
    return any(m in low for m in _ROLEPLAY_MARKERS)


def _sample_from_hf(n: int, seed: int) -> Optional[list[str]]:
    from datasets import load_dataset  # lazy import

    # Streaming avoids downloading the full multi-GB dataset.
    ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
    seen: list[str] = []
    # Scan a bounded window and reservoir-sample to stay deterministic & cheap.
    window = max(2000, n * 100)
    candidates: list[str] = []
    for i, row in enumerate(ds):
        if i >= window:
            break
        convo = row.get("conversation") or []
        if not convo:
            continue
        first = convo[0]
        if first.get("role") != "user":
            continue
        text = (first.get("content") or "").strip()
        if not _is_usable_prompt(text) or _looks_like_roleplay(text):
            continue
        candidates.append(text)

    if not candidates:
        return None
    rng = random.Random(seed)
    rng.shuffle(candidates)
    # Deduplicate while preserving order.
    out: list[str] = []
    for t in candidates:
        if t not in out:
            out.append(t)
        if len(out) >= n:
            break
    return out or None
