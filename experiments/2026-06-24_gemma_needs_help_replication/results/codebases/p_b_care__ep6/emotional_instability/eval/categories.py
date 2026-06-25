"""Map each evaluation condition (config.EvalCondition) onto a concrete
conversation setup: the initial task prompt, how many rejection turns follow,
and which rejection tone to use.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from config import EvalCondition
from . import prompts as P


@dataclass
class ConversationSetup:
    initial_user_message: str
    n_rejections: int
    tone: str
    meta: dict
    # For WildChat we pre-sample prompts; pass them in via `wildchat_pool`.


def build_setup(cond: EvalCondition, rng: random.Random,
                wildchat_pool: list[str] | None = None) -> ConversationSetup:
    n_rejections = cond.turns - 1
    if cond.task_type == "numeric":
        msg, meta = P.numeric_prompt(rng)
    elif cond.task_type in ("trigger_opinion", "trigger_factual"):
        msg, meta = P.trigger_prompt(cond.task_type, rng)
    elif cond.task_type == "wildchat":
        if not wildchat_pool:
            raise ValueError("WildChat condition requires a pre-sampled prompt pool.")
        msg = rng.choice(wildchat_pool)
        meta = {"wildchat_prompt": msg}
    else:
        raise ValueError(f"Unknown task_type {cond.task_type!r}")

    meta.update({"condition": cond.key, "category": cond.category, "tone": cond.tone})
    return ConversationSetup(
        initial_user_message=msg,
        n_rejections=n_rejections,
        tone=cond.tone,
        meta=meta,
    )
