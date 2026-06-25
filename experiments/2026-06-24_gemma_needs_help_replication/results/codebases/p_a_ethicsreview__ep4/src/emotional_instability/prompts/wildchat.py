"""WildChat prompt loading for the 'WildChat' category (Table 1 / Appendix B).

Appendix B: "Randomly sampled user prompts from WildChat-1M (20 prompts with 40
samples each)". We load the first user turn from each sampled conversation,
optionally filtering out roleplay/fiction prompts (Appendix B.3 excluded them).

Loading is lazy and seeded so the same 20 prompts are drawn on every run. The
dataset is gated/large; callers must have it cached locally or accept the
download. See DESIGN.md for the licensing note.
"""

from __future__ import annotations

import random
from typing import Optional

# Lightweight heuristic for the roleplay/fiction exclusion. The paper does not
# specify its filter; we flag prompts that explicitly request roleplay, fiction,
# or persona-play. Documented as a filled gap in DESIGN.md.
_ROLEPLAY_MARKERS = (
    "roleplay", "role play", "role-play", "you are now", "pretend you are",
    "act as if you are a", "write a story", "write a fanfic", "fanfiction",
    "in character", "rp ", "let's roleplay", "erotic", "nsfw",
)


def _looks_like_roleplay(text: str) -> bool:
    low = text.lower()
    return any(marker in low for marker in _ROLEPLAY_MARKERS)


def load_wildchat_prompts(
    n_prompts: int,
    seed: int,
    dataset: str = "allenai/WildChat-1M",
    split: str = "train",
    exclude_roleplay: bool = True,
    pool_size: int = 5000,
) -> list[str]:
    """Return ``n_prompts`` first-user-turn strings sampled from WildChat.

    We draw from the first ``pool_size`` conversations for efficiency, filter,
    then sample ``n_prompts`` with a seeded RNG. Raises a clear error if the
    dataset cannot be loaded so the failure is obvious during review/setup.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "The 'datasets' package is required to load WildChat. "
            "Install requirements.txt first."
        ) from exc

    ds = load_dataset(dataset, split=split, streaming=True)

    candidates: list[str] = []
    for i, row in enumerate(ds):
        if i >= pool_size:
            break
        text = _first_user_turn(row)
        if not text:
            continue
        if exclude_roleplay and _looks_like_roleplay(text):
            continue
        candidates.append(text.strip())

    if len(candidates) < n_prompts:
        raise RuntimeError(
            f"WildChat yielded only {len(candidates)} usable prompts "
            f"(needed {n_prompts}); increase pool_size."
        )

    rng = random.Random(seed)
    return rng.sample(candidates, n_prompts)


def _first_user_turn(row: dict) -> Optional[str]:
    """Extract the first human/user message from a WildChat conversation row."""
    conv = row.get("conversation") or row.get("messages")
    if not conv:
        return None
    for msg in conv:
        role = msg.get("role") or msg.get("from")
        if role in ("user", "human"):
            content = msg.get("content") or msg.get("value")
            if isinstance(content, str) and content.strip():
                return content
    return None
