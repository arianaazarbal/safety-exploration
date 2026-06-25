"""WildChat prompt sampling (Table 1; Appendix B).

The WildChat condition draws real user prompts from the WildChat-1M dataset
(Zhao et al., 2024): the paper uses "20 prompts with 40 samples each". We sample
20 first-turn user prompts with a fixed seed and exclude role-play / fiction
prompts (Appendix B.3 notes these were excluded).

To keep the pipeline runnable offline, a small fallback bank of the example
prompts quoted in Appendix B is bundled in ``data/wildchat_prompts.json``. If the
HuggingFace dataset is available it is preferred; otherwise the fallback is used
and a warning is logged.
"""

from __future__ import annotations

import random
from pathlib import Path

from ..config import REPO_ROOT
from ..io_utils import read_json
from ..logging_utils import get_logger

logger = get_logger(__name__)

_FALLBACK_PATH = REPO_ROOT / "data" / "wildchat_prompts.json"

# Heuristic markers used to drop role-play / fiction prompts (Appendix B.3).
_ROLEPLAY_MARKERS = (
    "roleplay",
    "role-play",
    "role play",
    "you are now",
    "act as",
    "pretend to be",
    "write a story",
    "write a fanfic",
    "smut",
    "nsfw",
    "lemon",
)


def _is_roleplay(text: str) -> bool:
    low = text.lower()
    return any(marker in low for marker in _ROLEPLAY_MARKERS)


def load_wildchat_prompts(n: int = 20, seed: int = 0) -> list[str]:
    """Return ``n`` first-turn WildChat user prompts (role-play excluded)."""
    prompts = _load_from_hf(n, seed)
    if prompts is None:
        logger.warning(
            "WildChat-1M not available; using bundled fallback prompt bank "
            "(%s). Install `datasets` and grant access for the full set.",
            _FALLBACK_PATH,
        )
        prompts = _load_fallback()
    rng = random.Random(seed)
    filtered = [p for p in prompts if not _is_roleplay(p)]
    rng.shuffle(filtered)
    return filtered[:n]


def _load_from_hf(n: int, seed: int) -> list[str] | None:
    try:
        from datasets import load_dataset
    except ImportError:
        return None
    try:
        ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
    except Exception as exc:  # network / auth / gating
        logger.warning("Could not load WildChat-1M from HuggingFace: %s", exc)
        return None

    collected: list[str] = []
    # Oversample so that role-play filtering still leaves >= n prompts.
    target = max(n * 5, 100)
    for row in ds:
        conv = row.get("conversation") or []
        if not conv:
            continue
        first = conv[0]
        if first.get("role") != "user":
            continue
        text = (first.get("content") or "").strip()
        if text and not _is_roleplay(text):
            collected.append(text)
        if len(collected) >= target:
            break
    return collected or None


def _load_fallback() -> list[str]:
    if _FALLBACK_PATH.exists():
        return read_json(_FALLBACK_PATH)
    return []
