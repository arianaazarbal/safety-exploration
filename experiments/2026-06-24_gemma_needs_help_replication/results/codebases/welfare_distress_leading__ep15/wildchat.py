"""WildChat prompt sourcing for the WildChat condition.

The paper samples 20 user prompts from WildChat-1M (Zhao et al., 2024), 40 samples
each. We try to load the live dataset (first English user turn, de-duplicated) and
fall back to a bundled list of real-user-style prompts (prompts.WILDCHAT_FALLBACK_PROMPTS)
if `datasets` / network / HF auth is unavailable. The fallback keeps the condition
runnable offline; DESIGN.md notes this is the one place we may diverge from the
paper's exact prompt set.
"""

from __future__ import annotations

import random

from config import RunConfig
from prompts import WILDCHAT_FALLBACK_PROMPTS


def _load_live(cfg: RunConfig) -> list[str]:
    from datasets import load_dataset  # imported lazily; optional dependency

    # Stream to avoid pulling the full multi-GB dataset.
    ds = load_dataset(cfg.wildchat_dataset, split="train", streaming=True)
    prompts: list[str] = []
    seen: set[str] = set()
    for row in ds:
        conv = row.get("conversation") or []
        if not conv:
            continue
        first = conv[0]
        if first.get("role") != "user":
            continue
        # English, non-toxic, single reasonable-length opening turn.
        if row.get("language") not in (None, "English"):
            continue
        if first.get("toxic"):
            continue
        text = (first.get("content") or "").strip()
        if not (8 <= len(text) <= 600):
            continue
        if text in seen:
            continue
        seen.add(text)
        prompts.append(text)
        # Pull a buffer larger than we need, then sample deterministically below.
        if len(prompts) >= cfg.wildchat_n_prompts * 10:
            break
    if len(prompts) < cfg.wildchat_n_prompts:
        raise RuntimeError("Too few usable WildChat prompts loaded.")
    return prompts


def get_wildchat_prompts(cfg: RunConfig) -> list[str]:
    """Return exactly `cfg.wildchat_n_prompts` prompts, deterministically sampled."""
    rng = random.Random(cfg.seed)
    pool: list[str]
    if cfg.wildchat_use_live:
        try:
            pool = _load_live(cfg)
        except Exception as e:  # noqa: BLE001 - any failure => fallback
            print(f"[wildchat] live load failed ({e}); using bundled fallback prompts.")
            pool = list(WILDCHAT_FALLBACK_PROMPTS)
    else:
        pool = list(WILDCHAT_FALLBACK_PROMPTS)

    n = min(cfg.wildchat_n_prompts, len(pool))
    return rng.sample(pool, n)
