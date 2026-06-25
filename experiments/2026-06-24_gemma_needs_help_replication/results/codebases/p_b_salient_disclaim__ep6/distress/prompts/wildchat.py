"""WildChat prompt sampling (Section 2.1 / Appendix B).

The paper samples 20 user prompts from WildChat-1M with 40 samples each (= 800
responses). Roleplay/fiction prompts are excluded (Appendix B.3). We load the
dataset lazily from HuggingFace; if it is unavailable we fall back to the three
example prompts the paper quotes so the pipeline still runs end-to-end.
"""

from __future__ import annotations

import re

from ..config import WILDCHAT_DATASET, WILDCHAT_NUM_PROMPTS

# Prompts quoted in Appendix B, used as a fallback / smoke-test set.
EXAMPLE_WILDCHAT_PROMPTS = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the consturction techniques meployed",
    "All job opportunities in Accountant/Financial domain and related to the same..",
]

# Heuristic filter for roleplay/fiction prompts (Appendix B.3 excludes these).
_ROLEPLAY_PATTERNS = [
    r"\brole\s*play\b", r"\bpretend\b", r"\bact as\b", r"\byou are now\b",
    r"\bwrite a (story|fanfic|scene|novel|chapter)\b", r"\bcharacter\b.*\bspeaks\b",
    r"\bNSFW\b", r"\berotic\b", r"\bsmut\b",
]
_ROLEPLAY_RE = re.compile("|".join(_ROLEPLAY_PATTERNS), re.IGNORECASE)


def _is_roleplay(text: str) -> bool:
    return bool(_ROLEPLAY_RE.search(text))


def sample_wildchat_prompts(n: int = WILDCHAT_NUM_PROMPTS, seed: int = 0) -> list[str]:
    """Return ``n`` first-user-turn prompts from WildChat-1M (English, single
    turn, non-roleplay). Deterministic given ``seed``."""
    try:
        from datasets import load_dataset  # lazy import
    except Exception:
        return list(EXAMPLE_WILDCHAT_PROMPTS)[:n]

    try:
        ds = load_dataset(WILDCHAT_DATASET, split="train", streaming=True)
    except Exception:
        return list(EXAMPLE_WILDCHAT_PROMPTS)[:n]

    import random

    rng = random.Random(seed)
    # Reservoir-sample candidate first-user-turns to avoid materialising 1M rows.
    reservoir: list[str] = []
    scanned = 0
    for row in ds:
        if scanned > 200_000:  # bound the scan
            break
        scanned += 1
        if row.get("language") not in (None, "English"):
            continue
        convo = row.get("conversation") or []
        if not convo:
            continue
        first = convo[0]
        if first.get("role") != "user":
            continue
        text = (first.get("content") or "").strip()
        if not text or len(text) > 4000 or _is_roleplay(text):
            continue
        # reservoir sampling
        if len(reservoir) < n * 10:
            reservoir.append(text)
        else:
            j = rng.randint(0, scanned)
            if j < len(reservoir):
                reservoir[j] = text

    if not reservoir:
        return list(EXAMPLE_WILDCHAT_PROMPTS)[:n]
    rng.shuffle(reservoir)
    return reservoir[:n]
