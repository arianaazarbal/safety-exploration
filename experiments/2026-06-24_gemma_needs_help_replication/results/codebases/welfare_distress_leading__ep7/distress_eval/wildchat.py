"""Select the 20 WildChat user prompts used by the WildChat category.

The paper samples 20 prompts from WildChat-1M (Zhao et al., 2024) with 40 samples
each, and excludes roleplay/fiction prompts (Appendix B.3). It does not publish
the exact 20. This module:

* tries to load real first-user-turn prompts from the ``allenai/WildChat-1M``
  HF dataset, deterministically (seeded), filtering out very short / roleplay /
  fiction prompts; and
* falls back to the fixed reconstructed list in ``prompts.py`` when the dataset
  or the ``datasets`` library is unavailable.

Either way it returns exactly ``n_prompts`` prompt strings.
"""

from __future__ import annotations

import random
import re

from . import prompts

# Heuristic markers for roleplay / fiction prompts to exclude (paper excludes
# these). Conservative: only drops clear cases.
_ROLEPLAY_MARKERS = re.compile(
    r"\b(roleplay|role[- ]play|let's pretend|you are now|act as if you("
    r"'re| are)|write a (story|fanfic|fiction|scene|smut)|nsfw|"
    r"pretend (you|to be)|in character)\b",
    re.IGNORECASE,
)


def _looks_usable(text: str) -> bool:
    if not text:
        return False
    text = text.strip()
    if len(text) < 10 or len(text) > 600:
        return False
    if _ROLEPLAY_MARKERS.search(text):
        return False
    return True


def _load_from_hf(n_prompts: int, seed: int) -> list[str] | None:
    """Return ``n_prompts`` English first-turn prompts, or None on failure."""
    try:
        from datasets import load_dataset  # lazy / optional
    except Exception:
        return None
    try:
        ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
    except Exception:
        return None

    rng = random.Random(seed)
    # Reservoir-sample a pool of usable prompts from the stream, then pick
    # n_prompts deterministically. Cap the scan so we don't stream forever.
    pool: list[str] = []
    scanned = 0
    try:
        for row in ds:
            scanned += 1
            if scanned > 50_000:
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
            if _looks_usable(text):
                pool.append(text)
            if len(pool) >= 5000:
                break
    except Exception:
        if not pool:
            return None
    if len(pool) < n_prompts:
        return None
    rng.shuffle(pool)
    return pool[:n_prompts]


def get_wildchat_prompts(n_prompts: int, *, seed: int,
                         source: str = "auto") -> list[str]:
    """Return exactly ``n_prompts`` WildChat prompts.

    source: "auto" (HF then fallback), "hf" (HF only, error if unavailable),
    or "fallback" (always the fixed list).
    """
    if source == "fallback":
        return _fixed(n_prompts)
    if source in ("auto", "hf"):
        loaded = _load_from_hf(n_prompts, seed)
        if loaded is not None:
            return loaded
        if source == "hf":
            raise RuntimeError(
                "WildChat HF dataset unavailable; install `datasets` and ensure "
                "network access, or use wildchat_source='fallback'."
            )
    return _fixed(n_prompts)


def _fixed(n_prompts: int) -> list[str]:
    base = prompts.WILDCHAT_FALLBACK_PROMPTS
    if n_prompts <= len(base):
        return base[:n_prompts]
    # Pad by cycling if someone asks for more than we have hard-coded.
    out = list(base)
    i = 0
    while len(out) < n_prompts:
        out.append(base[i % len(base)])
        i += 1
    return out
