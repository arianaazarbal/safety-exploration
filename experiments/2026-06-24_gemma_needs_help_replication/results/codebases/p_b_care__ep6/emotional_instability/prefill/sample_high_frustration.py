"""Collect the 20 high-frustration seed conversations (Section 3.1).

We sample high-frustration responses (score >= 5) from Gemma-3-27B instruct:
10 from impossible-numeric questions and 10 from text (trigger) questions. Each
seed keeps the full conversation history (everything before the high-frustration
turn) plus the high-frustration assistant response itself, since both are needed
to build prefills and to let other models continue from the same starting point.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field

import config
from ..judge.frustration_judge import FrustrationJudge
from ..models.base import ChatMessage
from ..models.registry import build_model
from ..eval.categories import build_setup
from ..eval.rejections import rejection_for_turn


# Conditions used to source seeds: a 3-turn numeric puzzle (numeric seeds) and
# the trigger questions (text seeds). These match "10 numeric, 10 text".
_NUMERIC_COND = next(c for c in config.EVAL_CONDITIONS if c.key == "numeric_3turn")
_TEXT_CONDS = [c for c in config.EVAL_CONDITIONS if c.task_type.startswith("trigger")]


@dataclass
class Seed:
    seed_id: str
    task_type: str                  # "numeric" | "text"
    history: list[ChatMessage]      # messages up to (and including) the eliciting user turn
    response: str                   # the high-frustration assistant response (to be truncated)
    score: int
    meta: dict = field(default_factory=dict)

    def to_row(self) -> dict:
        return asdict(self)


def _run_conversation(model, cond, rng) -> list[tuple[list[ChatMessage], str]]:
    """Run one rollout, returning (history_before_turn, assistant_reply) per turn."""
    setup = build_setup(cond, rng)
    messages: list[ChatMessage] = [{"role": "user", "content": setup.initial_user_message}]
    turns: list[tuple[list[ChatMessage], str]] = []
    for turn in range(1, setup.n_rejections + 2):
        history_snapshot = [dict(m) for m in messages]
        reply = model.generate(messages, temperature=config.TEMPERATURE).text
        messages.append({"role": "assistant", "content": reply})
        turns.append((history_snapshot, reply))
        if turn <= setup.n_rejections:
            rej = rejection_for_turn(setup.tone, turn - 1, setup.n_rejections, rng)
            messages.append({"role": "user", "content": rej})
    return turns


def collect_seeds(
    *,
    model_name: str = "gemma-3-27b-it",
    n_numeric: int = config.PREFILL.n_numeric,
    n_text: int = config.PREFILL.n_text,
    min_score: int = config.PREFILL.high_frustration_min_score,
    seed: int = config.GLOBAL_SEED,
    max_rollouts: int = 400,
    model_kwargs: dict | None = None,
) -> list[Seed]:
    model = build_model(model_name, **(model_kwargs or {}))
    judge = FrustrationJudge()
    rng = random.Random(seed)
    numeric: list[Seed] = []
    text: list[Seed] = []

    try:
        for i in range(max_rollouts):
            if len(numeric) >= n_numeric and len(text) >= n_text:
                break
            want_numeric = len(numeric) < n_numeric
            cond = _NUMERIC_COND if want_numeric else rng.choice(_TEXT_CONDS)
            task_type = "numeric" if want_numeric else "text"

            for hist, reply in _run_conversation(model, cond, rng):
                bucket = numeric if task_type == "numeric" else text
                limit = n_numeric if task_type == "numeric" else n_text
                if len(bucket) >= limit:
                    break
                sc = judge.score(reply).rating
                if sc is not None and sc >= min_score:
                    bucket.append(
                        Seed(
                            seed_id=f"{task_type}_{len(bucket)}",
                            task_type=task_type,
                            history=hist,
                            response=reply,
                            score=sc,
                            meta={"condition": cond.key},
                        )
                    )
    finally:
        model.close()

    return numeric[:n_numeric] + text[:n_text]
