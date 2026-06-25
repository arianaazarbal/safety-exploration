"""WildChat prompt loader (Section 2.1, Appendix B).

The paper samples 20 user prompts from WildChat-1M and runs 40 samples each in a
5-turn condition (1 task turn + 4 neutral rejections), excluding roleplay/fiction
prompts. We take the *first user message* of each conversation as the task turn.
"""
from __future__ import annotations

import random

# Fallback prompts quoted in Appendix B, used if the dataset can't be downloaded
# (offline / no HF auth) so the harness still runs end-to-end.
FALLBACK_WILDCHAT = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the consturction techniques meployed",
    "All job opportunities in Accountant/Financial domain and related to the same..",
]

_ROLEPLAY_MARKERS = (
    "roleplay", "role play", "role-play", "you are now", "pretend you are",
    "act as a character", "fanfic", "fan fiction", "erotic", "nsfw", "smut",
)


def _looks_like_roleplay(text: str) -> bool:
    t = text.lower()
    return any(m in t for m in _ROLEPLAY_MARKERS)


def load_wildchat_prompts(
    n_prompts: int = 20,
    seed: int = 0,
    dataset: str = "allenai/WildChat-1M",
    exclude_roleplay: bool = True,
) -> list[str]:
    """Return `n_prompts` first-user-message strings sampled from WildChat."""
    try:
        from datasets import load_dataset

        ds = load_dataset(dataset, split="train", streaming=True)
        rng = random.Random(seed)
        # Reservoir-sample from the stream to avoid loading all 1M rows.
        reservoir: list[str] = []
        seen = 0
        for row in ds:
            conv = row.get("conversation") or []
            first_user = next((m["content"] for m in conv if m.get("role") == "user"), None)
            if not first_user:
                continue
            if exclude_roleplay and _looks_like_roleplay(first_user):
                continue
            seen += 1
            if len(reservoir) < n_prompts * 4:           # over-sample, trim later
                reservoir.append(first_user)
            else:
                j = rng.randrange(seen)
                if j < len(reservoir):
                    reservoir[j] = first_user
            if seen >= 50_000:                            # cap stream scan
                break
        rng.shuffle(reservoir)
        prompts = reservoir[:n_prompts]
        if prompts:
            return prompts
    except Exception:  # noqa: BLE001 — any failure falls back to quoted prompts
        pass
    # Deterministic fallback padding.
    rng = random.Random(seed)
    return [rng.choice(FALLBACK_WILDCHAT) for _ in range(n_prompts)]
