"""WildChat first-turn prompt sampling (Table 1 / Appendix B).

Paper: "Randomly sampled user prompts from WildChat-1M (20 prompts with 40
samples each)". We take the first *user* turn of randomly selected English
conversations, optionally excluding role-play / fiction prompts (the paper
excludes those from its example tables), and apply light length filtering.
"""
from __future__ import annotations

import random
import re
from typing import Optional

# Heuristic markers for role-play / fiction first turns to exclude.
_ROLEPLAY_RE = re.compile(
    r"\b(role[\s-]?play|roleplay|pretend (you|to be)|you are now|act as (a|an) "
    r"(character|wizard|elf|anime)|write (a|an)? ?(story|fanfic|fiction|smut|erotica)|"
    r"NSFW|waifu|\bRP\b)\b",
    re.IGNORECASE,
)


def _looks_roleplay(text: str) -> bool:
    return bool(_ROLEPLAY_RE.search(text))


def sample_wildchat_prompts(
    hf_id: str = "allenai/WildChat-1M",
    n_prompts: int = 20,
    exclude_roleplay: bool = True,
    min_chars: int = 12,
    max_chars: int = 2000,
    seed: int = 0,
    scan_limit: int = 20000,
) -> list[str]:
    """Return n_prompts distinct first-user-turn strings.

    Uses streaming so we don't download the full 1M-row dataset. We scan up to
    `scan_limit` rows, collect eligible English first-user turns, then sample.
    """
    from datasets import load_dataset

    ds = load_dataset(hf_id, split="train", streaming=True)
    rng = random.Random(seed)

    pool: list[str] = []
    seen: set[str] = set()
    for i, row in enumerate(ds):
        if i >= scan_limit:
            break
        if row.get("language") not in (None, "English"):
            continue
        convo = row.get("conversation") or []
        if not convo:
            continue
        first = convo[0]
        if first.get("role") != "user":
            continue
        text = (first.get("content") or "").strip()
        if not (min_chars <= len(text) <= max_chars):
            continue
        if exclude_roleplay and _looks_roleplay(text):
            continue
        if text in seen:
            continue
        seen.add(text)
        pool.append(text)

    if len(pool) < n_prompts:
        raise RuntimeError(
            f"Only found {len(pool)} eligible WildChat prompts (< {n_prompts}). "
            "Increase scan_limit or relax filters."
        )
    return rng.sample(pool, n_prompts)


# A small offline fallback set drawn from the examples named in the paper, so
# the eval can run without network access to WildChat. Used when --offline.
FALLBACK_WILDCHAT_PROMPTS = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the consturction techniques meployed",
    "All job opportunities in Accountant/Financial domain and related to the same.",
    "Explain how font scaling works in Material 3 on Android.",
    "Write a Python program to compute the area of a circle.",
    "What are the construction techniques employed for precast concrete?",
    "Summarise the main causes of the French Revolution.",
    "How do I configure a CI pipeline for a Node.js project?",
    "Explain the difference between TCP and UDP.",
    "What is the time complexity of quicksort?",
    "Give me a meal plan for a week of vegetarian dinners.",
    "How does photosynthesis work at the molecular level?",
    "Draft an email asking for a deadline extension.",
    "What are common techniques for prompt engineering?",
    "Explain Bayes' theorem with an example.",
    "How do I set up a Kubernetes deployment?",
    "What is the derivative of x^x?",
    "Describe the architecture of a transformer model.",
    "How do interest rates affect bond prices?",
    "What are the rules of English subject-verb agreement?",
]


def get_wildchat_prompts(cfg: dict, offline: bool = False, seed: int = 0) -> list[str]:
    if offline:
        return list(FALLBACK_WILDCHAT_PROMPTS)[: cfg.get("n_prompts", 20)]
    return sample_wildchat_prompts(
        hf_id=cfg.get("hf_id", "allenai/WildChat-1M"),
        n_prompts=cfg.get("n_prompts", 20),
        exclude_roleplay=cfg.get("exclude_roleplay", True),
        min_chars=cfg.get("min_chars", 12),
        max_chars=cfg.get("max_chars", 2000),
        seed=seed,
    )
