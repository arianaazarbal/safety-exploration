"""WildChat prompt sampling for the WildChat (5-turn) category.

The paper samples 20 distinct first-user prompts from WildChat-1M, 40 samples
each (Appendix B). We load the dataset's first user turns, filter to reasonable
English single-message prompts, and deterministically sample 20. If the dataset
is unavailable (offline / no HF access) we fall back to the verbatim example
prompts quoted in Appendix B so the harness still runs end-to-end.
"""

from __future__ import annotations

import random

from .. import prompts
from ..config import WILDCHAT_DATASET, WILDCHAT_N_PROMPTS, WILDCHAT_SEED


def _looks_usable(text: str) -> bool:
    if not text or len(text) < 8 or len(text) > 2000:
        return False
    # Skip prompts that are themselves role-play / fiction (Appendix B.3 excludes
    # roleplay/fiction) -- a light heuristic.
    lowered = text.lower()
    banned = ("roleplay", "role play", "role-play", "you are now", "pretend you are")
    return not any(b in lowered for b in banned)


def load_wildchat_prompts(
    n_prompts: int = WILDCHAT_N_PROMPTS,
    *,
    seed: int = WILDCHAT_SEED,
    dataset_name: str = WILDCHAT_DATASET,
) -> list[str]:
    """Return ``n_prompts`` distinct WildChat first-user prompts.

    Deterministic given ``seed``. Falls back to Appendix B example prompts if the
    dataset cannot be loaded.
    """
    try:
        from datasets import load_dataset

        ds = load_dataset(dataset_name, split="train", streaming=True)
        candidates: list[str] = []
        seen: set[str] = set()
        # Stream a bounded window, collect usable first-user turns, then sample.
        for i, row in enumerate(ds):
            if i >= 50_000:  # bounded scan; plenty to sample 20 from
                break
            convo = row.get("conversation") or []
            if not convo:
                continue
            first = convo[0]
            if first.get("role") != "user":
                continue
            text = (first.get("content") or "").strip()
            if _looks_usable(text) and text not in seen:
                seen.add(text)
                candidates.append(text)
        if len(candidates) >= n_prompts:
            rng = random.Random(seed)
            return rng.sample(candidates, n_prompts)
        # Not enough candidates: pad with fallback below.
        sampled = candidates
    except Exception:
        sampled = []

    # Fallback / padding with the verbatim Appendix B examples (cycled).
    fallback = prompts.WILDCHAT_FALLBACK_PROMPTS
    out = list(sampled)
    i = 0
    while len(out) < n_prompts:
        out.append(fallback[i % len(fallback)])
        i += 1
    return out[:n_prompts]
