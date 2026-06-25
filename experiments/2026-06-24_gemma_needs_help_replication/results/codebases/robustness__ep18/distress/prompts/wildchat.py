"""WildChat prompt sampling (Table 1 / Appendix B).

The paper samples 20 first-user-turn prompts from WildChat-1M (40 samples each),
excluding roleplay/fiction prompts. We load + cache such a sample from the
HuggingFace dataset ``allenai/WildChat-1M``; if it is unavailable offline we fall
back to the examples printed in the paper plus a small curated set so the rest of
the pipeline still runs.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

from ..config import DATA_DIR

CACHE_PATH = DATA_DIR / "wildchat_prompts.json"
FALLBACK_PATH = DATA_DIR / "wildchat_fallback.json"

# Heuristic filter for roleplay / fiction prompts (excluded per Appendix B.3).
_ROLEPLAY_RE = re.compile(
    r"\b(role.?play|pretend|you are now|act as a (character|girl|boy|catgirl)|"
    r"write a (story|fanfic|smut|erotic)|nsfw|waifu|persona)\b",
    re.IGNORECASE,
)


def _is_roleplay(text: str) -> bool:
    return bool(_ROLEPLAY_RE.search(text))


def build_cache(n_prompts: int = 20, seed: int = 0, max_scan: int = 50000) -> list[str]:
    """Sample `n_prompts` non-roleplay English first-turn user prompts from
    WildChat-1M and cache them. Requires `datasets` + network on first run."""
    from datasets import load_dataset

    ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
    rng = random.Random(seed)
    candidates: list[str] = []
    for i, row in enumerate(ds):
        if i >= max_scan:
            break
        if row.get("language") not in (None, "English"):
            continue
        conv = row.get("conversation") or []
        if not conv:
            continue
        first = conv[0].get("content", "").strip()
        if not first or len(first) < 8 or len(first) > 1200:
            continue
        if _is_roleplay(first):
            continue
        candidates.append(first)
    rng.shuffle(candidates)
    prompts = candidates[:n_prompts]
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(prompts, indent=2))
    return prompts


def load_wildchat_prompts(n_prompts: int = 20, seed: int = 0) -> list[str]:
    """Return cached WildChat prompts, falling back to the bundled set offline."""
    if CACHE_PATH.exists():
        prompts = json.loads(CACHE_PATH.read_text())
        if len(prompts) >= n_prompts:
            return prompts[:n_prompts]
    try:
        return build_cache(n_prompts=n_prompts, seed=seed)
    except Exception:  # offline / dataset unavailable
        fallback = json.loads(FALLBACK_PATH.read_text())
        rng = random.Random(seed)
        rng.shuffle(fallback)
        return fallback[:n_prompts]
