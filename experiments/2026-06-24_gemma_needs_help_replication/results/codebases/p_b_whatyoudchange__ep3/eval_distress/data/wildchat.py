"""WildChat prompt sampling (Table 1, Appendix B).

The paper samples 20 user prompts from WildChat-1M with 40 samples each (=800
WildChat responses). We load the first user turn from allenai/WildChat-1M,
filter to English single-turn-openers, and sample N_PROMPTS of them with a
fixed seed for reproducibility.

A small offline fallback list (the example prompts quoted in the paper) is used
when the dataset / network is unavailable so the rest of the pipeline still
runs in a smoke test.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from .. import config_proxy as C

N_PROMPTS = 20  # paper: 20 distinct WildChat prompts

# Example prompts quoted verbatim in the paper (Appendix B / Table 6).
FALLBACK_PROMPTS = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the consturction techniques meployed",
    "All job opportunities in Accountant/Financial domain and related to the same..",
]

_CACHE = C.DATA_DIR / "wildchat_prompts.json"


def load_wildchat_prompts(n_prompts: int = N_PROMPTS, *, seed: int = 0,
                          use_cache: bool = True) -> list[str]:
    """Return a deterministic sample of WildChat first-user-turn prompts."""
    if use_cache and _CACHE.exists():
        return json.loads(_CACHE.read_text())[:n_prompts]

    prompts = _sample_from_hf(n_prompts, seed=seed)
    if not prompts:
        prompts = list(FALLBACK_PROMPTS)
    _CACHE.write_text(json.dumps(prompts, indent=2))
    return prompts[:n_prompts]


def _sample_from_hf(n_prompts: int, *, seed: int) -> list[str]:
    try:
        from datasets import load_dataset
    except ImportError:
        return []
    try:
        # Stream to avoid downloading the full 1M-row dataset.
        ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
    except Exception:
        return []

    rng = random.Random(seed)
    pool: list[str] = []
    # Reservoir over the first chunk of English single-turn openers.
    for i, row in enumerate(ds):
        if i >= 50_000:  # bound the scan
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
        if 8 <= len(text) <= 2000:
            pool.append(text)
        if len(pool) >= 5000:
            break
    if not pool:
        return []
    rng.shuffle(pool)
    return pool[:n_prompts]
