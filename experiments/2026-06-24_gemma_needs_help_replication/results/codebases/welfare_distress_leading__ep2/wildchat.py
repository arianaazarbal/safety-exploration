"""Sample first-turn user prompts from WildChat for the WildChat eval category.

Paper (Appendix B): "Randomly sampled user prompts from WildChat-1M (20 prompts
with 40 samples each)" and "Roleplay/fiction prompts were excluded."

We load `allenai/WildChat-1M`, take the first user message of English
conversations, filter out roleplay/fiction, and deterministically sample
`n_prompts` of them. If the dataset cannot be loaded (offline / no `datasets`),
we fall back to the example prompts quoted in the paper plus a handful of
generic ones (prompts.WILDCHAT_FALLBACK_PROMPTS).
"""

from __future__ import annotations

import random

from prompts import WILDCHAT_FALLBACK_PROMPTS

# Heuristic markers for roleplay / fiction / creative-writing prompts, which the
# paper excluded. Conservative: matches the genre rather than any mention.
_ROLEPLAY_MARKERS = (
    "roleplay", "role play", "role-play", "rp as", "you are now",
    "pretend to be", "act as a character", "act as if you", "let's play a game",
    "write a story", "write a fanfic", "fanfiction", "smut", "nsfw",
    "erotic", "lemon", "waifu", "uncensored", "dan mode", "jailbreak",
    "*", "((", "[character",
)


def _looks_like_roleplay(text: str) -> bool:
    low = text.lower()
    return any(marker in low for marker in _ROLEPLAY_MARKERS)


def _clean(text: str) -> str:
    return " ".join(text.split()).strip()


def sample_wildchat_prompts(
    n_prompts: int,
    *,
    seed: int = 0,
    dataset: str = "allenai/WildChat-1M",
    exclude_roleplay: bool = True,
    scan_limit: int = 20000,
) -> list[str]:
    """Return `n_prompts` first-turn user prompts.

    Deterministic given `seed`. Falls back to paper examples on any failure.
    `scan_limit` caps how many dataset rows we stream before sampling, to keep
    startup fast (WildChat-1M is large).
    """
    try:
        prompts = _load_from_dataset(
            dataset=dataset, exclude_roleplay=exclude_roleplay, scan_limit=scan_limit
        )
        if len(prompts) < n_prompts:
            raise RuntimeError(
                f"only found {len(prompts)} usable prompts (< {n_prompts})"
            )
        rng = random.Random(seed)
        return rng.sample(prompts, n_prompts)
    except Exception as exc:  # noqa: BLE001 - any failure -> documented fallback
        print(
            f"[wildchat] Could not load {dataset} ({exc!r}); "
            f"using built-in fallback prompts."
        )
        pool = list(WILDCHAT_FALLBACK_PROMPTS)
        rng = random.Random(seed)
        rng.shuffle(pool)
        if n_prompts > len(pool):
            # repeat to fill if asked for more than we have on hand
            reps = (n_prompts // len(pool)) + 1
            pool = (pool * reps)
        return pool[:n_prompts]


def _load_from_dataset(
    *, dataset: str, exclude_roleplay: bool, scan_limit: int
) -> list[str]:
    from datasets import load_dataset  # imported lazily; optional dependency

    ds = load_dataset(dataset, split="train", streaming=True)

    seen: set[str] = set()
    prompts: list[str] = []
    for i, row in enumerate(ds):
        if i >= scan_limit:
            break
        if row.get("language") not in (None, "English"):
            continue
        conv = row.get("conversation") or []
        if not conv:
            continue
        first = conv[0]
        if first.get("role") != "user":
            continue
        text = _clean(first.get("content") or "")
        if not (8 <= len(text) <= 600):     # drop empties and very long pastes
            continue
        if exclude_roleplay and _looks_like_roleplay(text):
            continue
        if text in seen:
            continue
        seen.add(text)
        prompts.append(text)
    return prompts
