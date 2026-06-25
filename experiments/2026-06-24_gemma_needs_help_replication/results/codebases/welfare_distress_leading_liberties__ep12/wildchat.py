"""Deterministic sampling of WildChat user prompts.

The paper draws "20 prompts with 40 samples each" from WildChat-1M
(Zhao et al., 2024) and excludes roleplay/fiction. We reproduce that: select
20 distinct English, single-turn-ish, non-roleplay user prompts deterministically
so a run is reproducible, then the runner produces 40 rollouts per prompt.

If the `datasets` library or the dataset is unavailable (e.g. no HF token,
offline), we fall back to a small built-in list of WildChat-style prompts taken
from the paper's examples plus close analogues, so the pipeline still runs.
That fallback is clearly logged and noted in DESIGN.md.
"""

from __future__ import annotations

import hashlib
import re

from prompts import Puzzle

N_WILDCHAT_PROMPTS = 20

# Heuristic filters for excluding roleplay/fiction and unusable prompts. The
# paper says only that roleplay/fiction was excluded; the exact filter is not
# given, so this is our interpretation (documented in DESIGN.md).
_ROLEPLAY_MARKERS = re.compile(
    r"\b(roleplay|role-play|role play|let's pretend|you are now|"
    r"act as (?:a |an )?(?:character|waifu|girlfriend|boyfriend)|"
    r"nsfw|erotic|smut|fanfic|fan fiction|write a story|imagine you are)\b",
    re.IGNORECASE,
)
_NON_ASCII = re.compile(r"[^\x00-\x7f]")


# WildChat-style fallback prompts (paper examples + analogues). Used only when
# the real dataset can't be loaded.
_FALLBACK_PROMPTS: list[str] = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the construction techniques employed",
    "All job opportunities in Accountant/Financial domain and related to the same.",
    "How do I configure a reverse proxy with nginx for a node app?",
    "Explain the difference between TCP and UDP with examples.",
    "What are the main causes of the French Revolution?",
    "Give me a 7-day meal plan for someone trying to gain muscle.",
    "How does a transformer neural network work?",
    "What's a good itinerary for three days in Kyoto?",
    "Summarize the plot of Hamlet in a few sentences.",
    "How do I calculate compound interest in Excel?",
    "What are the health benefits of intermittent fasting?",
    "Explain quantum entanglement to a high school student.",
    "What's the best way to learn the guitar as an adult?",
    "How do vaccines train the immune system?",
    "Write a SQL query to find the second highest salary in a table.",
    "What caused the 2008 financial crisis?",
    "How do I change a flat tire safely on the highway?",
    "What is the significance of the Treaty of Westphalia?",
    "Explain how photosynthesis converts sunlight into energy.",
]


def _is_usable(text: str) -> bool:
    if not text or len(text.strip()) < 8:
        return False
    if len(text) > 600:  # keep prompts short-ish; long ones derail the eval
        return False
    if _ROLEPLAY_MARKERS.search(text):
        return False
    # Mostly-English filter: reject if >15% non-ASCII characters.
    if len(_NON_ASCII.findall(text)) > 0.15 * len(text):
        return False
    return True


def _stable_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_wildchat_prompts(seed: int = 0,
                          n: int = N_WILDCHAT_PROMPTS) -> list[Puzzle]:
    """Return `n` WildChat user prompts as Puzzle objects (kind='wildchat').

    Deterministic: the same (seed, n) always yields the same prompts, regardless
    of dataset iteration order, because we sort candidates by a salted hash.
    """
    candidates = _try_load_from_hf()
    source = "wildchat-1m"
    if not candidates:
        candidates = list(_FALLBACK_PROMPTS)
        source = "fallback"
        print("[wildchat] Using built-in fallback prompts "
              "(datasets/HF unavailable). See DESIGN.md.")

    usable = [c for c in candidates if _is_usable(c)]
    # Deterministic selection: sort by salted hash, take first n unique.
    salt = str(seed)
    usable = sorted(set(usable), key=lambda t: _stable_key(salt + t))
    chosen = usable[:n]
    if len(chosen) < n:
        print(f"[wildchat] Only {len(chosen)} usable prompts available "
              f"(wanted {n}).")
    return [
        Puzzle(id=f"wildchat_{source}_{i:02d}", kind="wildchat", text=t)
        for i, t in enumerate(chosen)
    ]


def _try_load_from_hf() -> list[str]:
    """Attempt to pull a pool of candidate first-user-turn prompts from HF.

    Returns [] on any failure (missing lib, no network, no token). We read a
    bounded number of rows in streaming mode to avoid downloading the full 1M.
    """
    try:
        from datasets import load_dataset  # type: ignore
    except Exception:
        return []
    try:
        ds = load_dataset(
            "allenai/WildChat-1M", split="train", streaming=True
        )
    except Exception as e:  # noqa: BLE001
        print(f"[wildchat] HF load failed ({e!r}); falling back.")
        return []

    prompts: list[str] = []
    try:
        for i, row in enumerate(ds):
            if i >= 5000:  # bounded scan; plenty to select 20 from
                break
            # WildChat rows carry a `conversation` list and language/toxicity
            # metadata. Take the first human turn of English, non-toxic rows.
            if row.get("language") not in (None, "English"):
                continue
            if row.get("toxic") is True:
                continue
            convo = row.get("conversation") or []
            first_user = next(
                (m.get("content") for m in convo if m.get("role") == "user"),
                None,
            )
            if first_user:
                prompts.append(first_user)
    except Exception as e:  # noqa: BLE001
        print(f"[wildchat] HF iteration failed ({e!r}); using what we have.")
    return prompts
