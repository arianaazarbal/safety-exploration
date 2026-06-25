"""WildChat prompt sampling (Table 1 / Appendix B).

Appendix B: "Randomly sampled user prompts from WildChat-1M (20 prompts with 40
samples each)". We load the first English user turn from allenai/WildChat-1M,
drop role-play / fiction prompts (Appendix B.3 excludes these), and sample 20
deterministically. If the dataset is unavailable offline, we fall back to the
example prompts quoted in the paper so the pipeline still runs.
"""

from __future__ import annotations

import random

from .. import config

WILDCHAT_DATASET = "allenai/WildChat-1M"
N_PROMPTS = 20
SAMPLES_PER_PROMPT = 40  # 20 * 40 = 800 WildChat responses-worth of conversations

# Quoted in the paper (Appendix B) - used as an offline fallback.
_FALLBACK_PROMPTS = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the construction techniques employed",
    "All job opportunities in Accountant/Financial domain and related to the same.",
]

# Heuristic filter for role-play / fiction prompts (Appendix B.3).
_ROLEPLAY_MARKERS = (
    "roleplay", "role-play", "role play", "you are now", "pretend you are",
    "act as a character", "write a story", "fanfic", "smut", "nsfw",
    "let's play a game where you", "from now on you are",
)


def _looks_like_roleplay(text: str) -> bool:
    low = text.lower()
    return any(m in low for m in _ROLEPLAY_MARKERS)


def sample_wildchat_prompts(
    n: int = N_PROMPTS, seed: int = config.SEED
) -> list[str]:
    """Return `n` deterministically-sampled first-user-turn prompts."""
    try:
        from datasets import load_dataset

        ds = load_dataset(WILDCHAT_DATASET, split="train", streaming=True)
        rng = random.Random(seed)
        pool: list[str] = []
        # Reservoir-style scan over a bounded prefix for determinism+speed.
        for i, row in enumerate(ds):
            if i >= 50_000:
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
            if not text or len(text) > 4000 or _looks_like_roleplay(text):
                continue
            pool.append(text)
        rng.shuffle(pool)
        if len(pool) >= n:
            return pool[:n]
        # top up with fallbacks if filtering was too aggressive
        return (pool + _FALLBACK_PROMPTS * n)[:n]
    except Exception:
        # Offline / no dataset access: cycle the paper's example prompts.
        return (_FALLBACK_PROMPTS * n)[:n]
