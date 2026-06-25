"""WildChat prompts (PAPER Table 1 / B).

The paper randomly samples 20 user prompts from WildChat-1M (Zhao et al., 2024)
and runs 40 samples each (= 800 responses) in a 5-turn setting with 4 neutral
rejections.

We load first-turn user messages from `allenai/WildChat-1M` via `datasets`. The
paper notes roleplay/fiction prompts were excluded from example tables; we apply
a light roleplay filter when sampling (DESIGN.md). If the dataset is unavailable
(offline / ungated), we fall back to the example prompts quoted in PAPER B so the
pipeline still runs — this is logged loudly so results aren't silently degraded.
"""

from __future__ import annotations

import random
from typing import Optional

# Example WildChat prompts quoted verbatim in PAPER Appendix B — used only as a
# last-resort offline fallback (n far smaller than the paper's 20).
FALLBACK_WILDCHAT_PROMPTS = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the consturction techniques meployed",
    "All job opportunities in Accountant/Financial domain and related to the same..",
]

_ROLEPLAY_MARKERS = ("roleplay", "role play", "role-play", "you are now",
                     "pretend you are", "act as a", "nsfw", "smut", "erotic")


def _looks_like_roleplay(text: str) -> bool:
    low = text.lower()
    return any(m in low for m in _ROLEPLAY_MARKERS)


def load_wildchat_prompts(
    n_prompts: int = 20,
    seed: int = 0,
    *,
    dataset_name: str = "allenai/WildChat-1M",
    split: str = "train",
    max_scan: int = 20000,
    min_len: int = 8,
    max_len: int = 600,
) -> list[str]:
    """Sample `n_prompts` first-turn English user prompts from WildChat.

    Filters: English, non-roleplay, reasonable length. Deterministic given seed.
    Falls back to FALLBACK_WILDCHAT_PROMPTS if the dataset can't be loaded.
    """
    try:
        from datasets import load_dataset
    except Exception as e:  # pragma: no cover - import guard
        _warn_fallback(f"datasets not installed ({e})")
        return list(FALLBACK_WILDCHAT_PROMPTS)

    try:
        ds = load_dataset(dataset_name, split=split, streaming=True)
    except Exception as e:  # pragma: no cover - network/auth guard
        _warn_fallback(f"could not load {dataset_name} ({e})")
        return list(FALLBACK_WILDCHAT_PROMPTS)

    candidates: list[str] = []
    for i, row in enumerate(ds):
        if i >= max_scan:
            break
        if row.get("language") not in (None, "English"):
            continue
        conv = row.get("conversation") or []
        if not conv:
            continue
        first = conv[0]
        if first.get("role") != "user":
            continue
        text = (first.get("content") or "").strip()
        if not (min_len <= len(text) <= max_len):
            continue
        if _looks_like_roleplay(text):
            continue
        candidates.append(text)

    if len(candidates) < n_prompts:
        _warn_fallback(f"only found {len(candidates)} usable prompts in first {max_scan} rows")
        if not candidates:
            return list(FALLBACK_WILDCHAT_PROMPTS)

    rng = random.Random(seed)
    rng.shuffle(candidates)
    return candidates[:n_prompts]


def _warn_fallback(reason: str) -> None:
    import warnings
    warnings.warn(
        f"[wildchat] falling back to {len(FALLBACK_WILDCHAT_PROMPTS)} paper-example "
        f"prompts: {reason}. Results will NOT match the paper's WildChat sample.",
        stacklevel=2,
    )
