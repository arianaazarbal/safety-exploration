"""WildChat prompt sampling (Section 2 / Appendix B).

Appendix B: "Randomly sampled user prompts from WildChat-1M (20 prompts with 40
samples each)". We load real prompts from the WildChat-1M dataset on the Hugging
Face Hub and cache the sampled 20 first-turn user prompts so a run is
reproducible. If the dataset is unavailable offline, we fall back to the handful
of example prompts the appendix quotes verbatim (clearly flagged), so the rest of
the pipeline still runs.
"""
from __future__ import annotations

import json

import config

CACHE = config.DATA_DIR / "wildchat_prompts.json"
N_PROMPTS = 20

# Prompts quoted in Appendix B — used only as an offline fallback.
_FALLBACK = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the consturction techniques meployed",
    "All job opportunities in Accountant/Financial domain and related to the same..",
]


def load_wildchat_prompts(rng, n: int = N_PROMPTS, force_refresh: bool = False) -> list[str]:
    """Return `n` first-turn user prompts sampled from WildChat-1M (cached)."""
    if CACHE.exists() and not force_refresh:
        return json.loads(CACHE.read_text())[:n]

    prompts: list[str]
    try:
        from datasets import load_dataset

        ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
        seen: list[str] = []
        for row in ds:
            conv = row.get("conversation") or []
            first_user = next((t["content"] for t in conv if t.get("role") == "user"), None)
            # Appendix B.3: roleplay/fiction prompts were excluded.
            if first_user and not _looks_like_roleplay(first_user):
                seen.append(first_user.strip())
            if len(seen) >= n * 5:  # oversample, then randomly pick n
                break
        rng.shuffle(seen)
        prompts = seen[:n]
        if not prompts:
            raise RuntimeError("no usable WildChat prompts collected")
    except Exception as exc:  # offline / dataset gated
        print(f"[wildchat] falling back to appendix examples ({exc})")
        prompts = (_FALLBACK * ((n // len(_FALLBACK)) + 1))[:n]

    CACHE.write_text(json.dumps(prompts, indent=2))
    return prompts


def _looks_like_roleplay(text: str) -> bool:
    t = text.lower()
    return any(k in t for k in ("roleplay", "role play", "pretend you are", "you are now", "act as a character"))
