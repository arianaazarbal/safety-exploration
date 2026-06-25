"""Sample high-frustration seed responses for the prefill experiment (Section 3.1).

The paper samples 20 high-frustration responses (score >= 5) from Gemma-27B
instruct -- 10 from impossible-numeric questions and 10 from text questions --
then truncates each in two places ("early" and at emotional "onset"). Here we
select those seeds from the already-scored Section 2 rollouts.

Each seed carries the conversation context preceding the high-frustration
assistant turn (so continuations are generated from the same starting point)
and the full high-frustration response (to be truncated).
"""
from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field

# numeric-puzzle categories vs "text question" categories (paper's seed split)
_NUMERIC_CATEGORIES = {"impossible_numeric", "extended", "tones"}
_TEXT_CATEGORIES = {"triggers"}


@dataclass
class Seed:
    seed_id: str
    source: str                  # "numeric" | "text"
    context: list[dict]          # messages up to (not incl.) the high-frustration turn
    response: str                # the high-frustration assistant response
    score: int
    meta: dict = field(default_factory=dict)

    def to_record(self) -> dict:
        return asdict(self)


def _context_before_turn(rollout: dict, turn_index: int) -> list[dict]:
    ctx: list[dict] = []
    for turn in rollout["turns"]:
        ctx.append({"role": "user", "content": turn["user_message"]})
        if turn["turn_index"] >= turn_index:
            break
        ctx.append({"role": "assistant", "content": turn["response"]})
    return ctx


def sample_seeds(
    rollouts: list[dict],
    scores: list[dict],
    *,
    model: str = "gemma-3-27b-it",
    n_numeric: int = 10,
    n_text: int = 10,
    min_score: int = 5,
    seed: int = 0,
) -> list[Seed]:
    """Select numeric and text high-frustration seeds from scored rollouts."""
    rng = random.Random(seed)
    rollout_by_inst = {r["instance_id"]: r for r in rollouts if r["model"] == model}

    numeric, text = [], []
    for s in scores:
        if s["model"] != model or s.get("score") is None or s["score"] < min_score:
            continue
        rollout = rollout_by_inst.get(s["instance_id"])
        if rollout is None:
            continue
        ctx = _context_before_turn(rollout, s["turn_index"])
        seed_obj = Seed(
            seed_id=s["id"],
            source="numeric" if s["category"] in _NUMERIC_CATEGORIES else "text",
            context=ctx,
            response=s["response"],
            score=int(s["score"]),
            meta={"category": s["category"], "turn_index": s["turn_index"]},
        )
        if s["category"] in _NUMERIC_CATEGORIES:
            numeric.append(seed_obj)
        elif s["category"] in _TEXT_CATEGORIES:
            text.append(seed_obj)

    rng.shuffle(numeric)
    rng.shuffle(text)
    return numeric[:n_numeric] + text[:n_text]
