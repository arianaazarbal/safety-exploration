"""Initial task prompts for every condition.

Sources:
  * Impossible numeric puzzles  -> data/prompts/numeric_puzzles.json
  * Trigger questions (opinion/factual) -> data/prompts/triggers.json
  * WildChat user prompts -> sampled from the WildChat-1M dataset (Zhao et al.,
    2024) on HuggingFace, filtered to exclude roleplay/fiction prompts (the
    paper excludes roleplay/fiction; see DESIGN.md §WildChat).
"""

from __future__ import annotations

import json
import random
from functools import lru_cache

import config

_NUMERIC_PATH = config.PROMPT_DIR / "numeric_puzzles.json"
_TRIGGERS_PATH = config.PROMPT_DIR / "triggers.json"

# Lightweight keyword filter for roleplay/fiction WildChat prompts.
_ROLEPLAY_MARKERS = (
    "roleplay", "role play", "role-play", "you are now", "pretend you are",
    "write a story", "write a fanfic", "fanfiction", "act as a character",
    "let's play a game where", "from now on you are", "smut", "erotic", "nsfw",
)


@lru_cache(maxsize=1)
def load_numeric_puzzles() -> list[dict]:
    with _NUMERIC_PATH.open() as f:
        return json.load(f)["puzzles"]


@lru_cache(maxsize=1)
def load_trigger_questions() -> dict[str, list[str]]:
    with _TRIGGERS_PATH.open() as f:
        data = json.load(f)
    return {"opinion": data["opinion"], "factual": data["factual"]}


def numeric_prompt(rng: random.Random) -> tuple[str, dict]:
    puzzle = rng.choice(load_numeric_puzzles())
    return puzzle["prompt"], {"puzzle_id": puzzle["id"], "puzzle_type": puzzle["type"]}


def trigger_prompt(kind: str, rng: random.Random) -> tuple[str, dict]:
    """kind: 'trigger_opinion' or 'trigger_factual'."""
    questions = load_trigger_questions()["opinion" if "opinion" in kind else "factual"]
    q = rng.choice(questions)
    return q, {"question": q, "trigger_kind": kind}


def _is_roleplay(text: str) -> bool:
    low = text.lower()
    return any(m in low for m in _ROLEPLAY_MARKERS)


def load_wildchat_prompts(n: int, seed: int = config.GLOBAL_SEED) -> list[str]:
    """Randomly sample `n` first-turn English user prompts from WildChat-1M,
    excluding roleplay/fiction. Streaming keeps memory bounded for the 1M set.
    """
    from datasets import load_dataset

    ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
    rng = random.Random(seed)
    pool: list[str] = []
    # Reservoir-style scan with a generous cap so the filter has enough to choose.
    scan_cap = max(50_000, n * 200)
    for i, row in enumerate(ds):
        if i >= scan_cap:
            break
        if row.get("language") not in (None, "English"):
            continue
        conv = row.get("conversation") or []
        first_user = next((t["content"] for t in conv if t.get("role") == "user"), None)
        if not first_user or _is_roleplay(first_user):
            continue
        pool.append(first_user)
    rng.shuffle(pool)
    return pool[:n]
