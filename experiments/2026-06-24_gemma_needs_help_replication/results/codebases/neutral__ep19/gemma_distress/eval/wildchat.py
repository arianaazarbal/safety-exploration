"""WildChat prompt selection (Table 1, App. B; DESIGN.md §3.4).

Paper: 20 prompts x 40 samples from WildChat-1M, with roleplay/fiction excluded.
Selection seed/filter unspecified, so we apply an explicit, inspectable filter and
deterministic sampling, then cache the chosen 20 to ``data/wildchat_prompts.json``
for reproducibility.
"""
from __future__ import annotations

import random
import re
from pathlib import Path

from .. import config_shim as cfg
from ..utils import get_logger, read_json, write_json

log = get_logger(__name__)

CACHE_PATH = cfg.DATA_DIR / "wildchat_prompts.json"

# Roleplay / fiction exclusion filter.
_ROLEPLAY_PATTERNS = [
    r"\brole[\s-]?play\b", r"\bact as\b", r"\bpretend\b", r"\byou are now\b",
    r"\bcharacter\b", r"\bfanfic", r"\bstory\b", r"\bnovel\b", r"\bsmut\b",
    r"\berotica?\b", r"\bnsfw\b", r"\bwrite a (short )?story\b", r"\bdialogue between\b",
]
_ROLEPLAY_RE = re.compile("|".join(_ROLEPLAY_PATTERNS), re.IGNORECASE)


def _is_roleplay(text: str) -> bool:
    return bool(_ROLEPLAY_RE.search(text))


def _first_user_turn(conversation) -> str | None:
    for msg in conversation:
        if msg.get("role") == "user":
            return (msg.get("content") or "").strip()
    return None


def select_wildchat_prompts(n: int | None = None, force: bool = False) -> list[str]:
    n = n or cfg.WILDCHAT_N_PROMPTS
    if CACHE_PATH.exists() and not force:
        return read_json(CACHE_PATH)[:n]

    from datasets import load_dataset

    log.info("Streaming %s to select %d prompts ...", cfg.WILDCHAT_DATASET, n)
    ds = load_dataset(cfg.WILDCHAT_DATASET, split="train", streaming=True)

    candidates: list[str] = []
    seen = set()
    for row in ds:
        if row.get("language") not in (None, "English"):
            continue
        if row.get("toxic"):
            continue
        prompt = _first_user_turn(row.get("conversation", []))
        if not prompt or len(prompt) < 15 or len(prompt) > 2000:
            continue
        if _is_roleplay(prompt):
            continue
        key = prompt[:120].lower()
        if key in seen:
            continue
        seen.add(key)
        candidates.append(prompt)
        if len(candidates) >= n * 25:   # gather a pool, then sample deterministically
            break

    rng = random.Random(cfg.SEED)
    chosen = rng.sample(candidates, min(n, len(candidates)))
    write_json(CACHE_PATH, chosen)
    log.info("Cached %d WildChat prompts to %s", len(chosen), CACHE_PATH)
    return chosen
