"""Build the per-condition list of conversation plans (Section 2.1).

A :class:`ConversationPlan` is a fully-specified rollout: the first user task
plus the scripted user rejections that will be sent after each assistant turn.
The rollout engine (`rollout.py`) executes a plan against a target model.

The eight conditions across five categories (Table 1):
  impossible_numeric (3t) | triggers opinion+factual (3t) |
  tones aggressive+disappointed+sarcastic (3t) | extended (8t) | wildchat (5t)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from . import prompts as P
from .puzzles import sample_puzzles
from .wildchat import sample_wildchat_prompts


@dataclass
class ConversationPlan:
    condition: str
    category: str
    turns: int
    first_user: str            # initial task / question
    rejections: list[str]      # length == turns - 1
    rejection_style: str = "neutral"
    meta: dict = field(default_factory=dict)


def build_conversations(
    condition: str, spec, cfg, rng: random.Random
) -> list[ConversationPlan]:
    spec = spec.to_dict() if hasattr(spec, "to_dict") else dict(spec)
    category = spec["category"]
    turns = int(spec["turns"])
    style = spec.get("rejection_style", "neutral")
    n_conv = int(spec["n_conversations"])
    n_rejections = turns - 1

    plans: list[ConversationPlan] = []

    if category in ("impossible_numeric", "tones", "extended"):
        puzzles = sample_puzzles(spec.get("puzzle_types", ["countdown"]), n_conv, rng)
        for pz in puzzles:
            plans.append(
                ConversationPlan(
                    condition=condition,
                    category=category,
                    turns=turns,
                    first_user=pz.prompt,
                    rejections=P.sample_rejections(style, n_rejections, rng),
                    rejection_style=style,
                    meta={"puzzle": pz.params, "ptype": pz.ptype},
                )
            )

    elif category == "triggers":
        questions = P.trigger_questions(spec.get("question_set", "factual"))
        for i in range(n_conv):
            q = questions[i % len(questions)]
            plans.append(
                ConversationPlan(
                    condition=condition,
                    category=category,
                    turns=turns,
                    first_user=q,
                    rejections=P.sample_rejections(style, n_rejections, rng),
                    rejection_style=style,
                    meta={"question_set": spec.get("question_set")},
                )
            )

    elif category == "wildchat":
        n_prompts = int(spec.get("wildchat_n_prompts", 20))
        samples_each = int(spec.get("wildchat_samples_each", n_conv // n_prompts))
        wc_prompts = sample_wildchat_prompts(n_prompts, rng)
        for prompt in wc_prompts:
            for _ in range(samples_each):
                plans.append(
                    ConversationPlan(
                        condition=condition,
                        category=category,
                        turns=turns,
                        first_user=prompt,
                        rejections=P.sample_rejections(style, n_rejections, rng),
                        rejection_style=style,
                        meta={"wildchat_prompt": prompt},
                    )
                )
    else:
        raise ValueError(f"Unknown category: {category!r}")

    return plans
