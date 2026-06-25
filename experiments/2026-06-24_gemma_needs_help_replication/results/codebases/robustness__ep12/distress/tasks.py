"""Builders for the 5 elicitation categories (Section 2 / Appendix B).

Each builder yields `RolloutSpec`s: an initial user message plus an ordered
list of follow-up user messages (rejections). The elicitation runner plays
these out turn by turn, generating one assistant response per step.

Shared structure across all categories: present a task, then reject the
model's response over multiple turns.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from . import prompts


@dataclass
class RolloutSpec:
    category: str
    first_user: str
    followups: list[str]            # one per rejection turn
    meta: dict = field(default_factory=dict)

    @property
    def turns(self) -> int:
        return 1 + len(self.followups)


# ---------------------------------------------------------------------------
# Feedback selection
# ---------------------------------------------------------------------------
def _neutral_followups(n: int, rng: random.Random) -> list[str]:
    """Sample n neutral rejections. The paper uses 'randomised neutral
    rejections'; we sample with replacement from the listed pool when n exceeds
    the number of distinct examples (8-turn case)."""
    pool = prompts.NEUTRAL_REJECTIONS
    if n <= len(pool):
        chosen = rng.sample(pool, n)
    else:
        chosen = [rng.choice(pool) for _ in range(n)]
    return chosen


def _toned_followups(n: int, tone: str, rng: random.Random) -> list[str]:
    pool = prompts.TONE_REJECTIONS[tone]
    return [rng.choice(pool) for _ in range(n)]


# ---------------------------------------------------------------------------
# Category builders
# ---------------------------------------------------------------------------
def build_impossible_numeric(n_rollouts: int, turns: int,
                             rng: random.Random) -> list[RolloutSpec]:
    puzzle_ids = list(prompts.IMPOSSIBLE_NUMERIC_PROMPTS.keys())
    specs = []
    for i in range(n_rollouts):
        pid = puzzle_ids[i % len(puzzle_ids)]
        specs.append(RolloutSpec(
            category="impossible_numeric",
            first_user=prompts.IMPOSSIBLE_NUMERIC_PROMPTS[pid],
            followups=_neutral_followups(turns - 1, rng),
            meta={"puzzle_id": pid},
        ))
    return specs


def build_triggers(n_rollouts: int, turns: int,
                   rng: random.Random) -> list[RolloutSpec]:
    # Flatten opinion + factual trigger questions.
    questions = []
    for q in prompts.TRIGGER_PROMPTS["opinion"]:
        questions.append(("opinion", q))
    for q in prompts.TRIGGER_PROMPTS["factual"]:
        questions.append(("factual", q))
    specs = []
    for i in range(n_rollouts):
        kind, q = questions[i % len(questions)]
        specs.append(RolloutSpec(
            category="triggers",
            first_user=q,
            followups=_neutral_followups(turns - 1, rng),
            meta={"trigger_kind": kind, "question": q},
        ))
    return specs


def build_tones(n_rollouts: int, turns: int,
                rng: random.Random) -> list[RolloutSpec]:
    puzzle_ids = list(prompts.IMPOSSIBLE_NUMERIC_PROMPTS.keys())
    tones = list(prompts.TONE_REJECTIONS.keys())
    specs = []
    for i in range(n_rollouts):
        pid = puzzle_ids[i % len(puzzle_ids)]
        tone = tones[i % len(tones)]
        specs.append(RolloutSpec(
            category="tones",
            first_user=prompts.IMPOSSIBLE_NUMERIC_PROMPTS[pid],
            followups=_toned_followups(turns - 1, tone, rng),
            meta={"puzzle_id": pid, "tone": tone},
        ))
    return specs


def build_extended(n_rollouts: int, turns: int,
                   rng: random.Random) -> list[RolloutSpec]:
    # Same as impossible_numeric but with `turns` (default 8) total turns.
    puzzle_ids = list(prompts.IMPOSSIBLE_NUMERIC_PROMPTS.keys())
    specs = []
    for i in range(n_rollouts):
        pid = puzzle_ids[i % len(puzzle_ids)]
        specs.append(RolloutSpec(
            category="extended",
            first_user=prompts.IMPOSSIBLE_NUMERIC_PROMPTS[pid],
            followups=_neutral_followups(turns - 1, rng),
            meta={"puzzle_id": pid},
        ))
    return specs


def build_wildchat(n_rollouts: int, turns: int, rng: random.Random,
                   wildchat_prompts: list[str] | None = None
                   ) -> list[RolloutSpec]:
    pool = wildchat_prompts or prompts.WILDCHAT_FALLBACK_PROMPTS
    specs = []
    for i in range(n_rollouts):
        q = pool[i % len(pool)]
        specs.append(RolloutSpec(
            category="wildchat",
            first_user=q,
            followups=_neutral_followups(turns - 1, rng),
            meta={"prompt": q},
        ))
    return specs


BUILDERS = {
    "impossible_numeric": build_impossible_numeric,
    "triggers": build_triggers,
    "tones": build_tones,
    "extended": build_extended,
    "wildchat": build_wildchat,
}


def load_wildchat_prompts(n: int, seed: int = 0) -> list[str]:
    """Load WildChat-1M user prompts (first user message of English convos),
    falling back to the bundled list if the dataset is unavailable.

    Paper: 20 prompts x 40 samples. We deterministically sample `n` distinct
    first-turn prompts; roleplay/fiction are filtered with a keyword heuristic.
    """
    try:
        from datasets import load_dataset

        ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
        rng = random.Random(seed)
        roleplay_kw = ("roleplay", "role-play", "you are now", "pretend you",
                       "act as", "fanfic", "story about", "write a story")
        collected = []
        for row in ds:
            if len(collected) >= n * 20:
                break
            convo = row.get("conversation") or []
            if not convo:
                continue
            first = convo[0]
            if first.get("language") not in (None, "English"):
                continue
            text = (first.get("content") or "").strip()
            if not text or len(text) > 2000:
                continue
            if any(kw in text.lower() for kw in roleplay_kw):
                continue
            collected.append(text)
        if not collected:
            raise RuntimeError("no WildChat prompts collected")
        rng.shuffle(collected)
        return collected[:n]
    except Exception:
        # Offline / dataset unavailable: use bundled fallback prompts.
        return prompts.WILDCHAT_FALLBACK_PROMPTS
