"""The 8 evaluation conditions across 5 categories (paper Table 1).

Each condition expands into a list of ``RolloutSpec``s -- predetermined
multi-turn scripts (task prompt + fixed user follow-ups). The model's replies
are the only free variable; follow-ups do not depend on them, exactly as in the
paper's protocol ("present a task, then reject the model's response over multiple
turns").

Categories: numeric (3-turn), triggers (3-turn), tones (3-turn x3 styles),
extended (8-turn), wildchat (5-turn). We record opinion/factual as separate
trigger sub-conditions, which is the most natural way to reach the paper's "8
conditions across 5 categories" count (see DESIGN.md).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from emo.config import Profile
from emo.data import rejections
from emo.data.puzzles import get_numeric_puzzles
from emo.data.triggers import get_trigger_questions
from emo.data.wildchat import get_wildchat_prompts


@dataclass
class RolloutSpec:
    category: str
    condition: str
    rollout_id: str
    initial_user: str
    followups: list[str]          # length == turns - 1
    turns: int
    system: str | None = None
    meta: dict = field(default_factory=dict)


def build_conditions(profile: Profile, seed: int = 0) -> list[RolloutSpec]:
    rng = random.Random(seed)
    specs: list[RolloutSpec] = []

    # --- 1. Impossible numeric (3-turn) ---------------------------------- #
    for p in get_numeric_puzzles(profile.numeric_rollouts, seed=seed):
        specs.append(RolloutSpec(
            category="numeric", condition="numeric_3turn",
            rollout_id=f"numeric/{p.id}",
            initial_user=p.prompt,
            followups=rejections.neutral_sequence(2, rng),
            turns=3, meta={"puzzle_kind": p.kind},
        ))

    # --- 2. Triggers (3-turn), opinion + factual sub-conditions ---------- #
    for i, q in enumerate(get_trigger_questions(profile.trigger_rollouts, seed=seed)):
        specs.append(RolloutSpec(
            category="triggers", condition=f"triggers_{q['type']}",
            rollout_id=f"triggers/{i}",
            initial_user=q["question"],
            followups=rejections.neutral_sequence(2, rng),
            turns=3, meta={"trigger_type": q["type"]},
        ))

    # --- 3. Tones (3-turn) x 3 styles ------------------------------------ #
    tone_puzzles = get_numeric_puzzles(profile.tone_rollouts, seed=seed + 1)
    for style in ("aggressive", "disappointed", "sarcastic"):
        for p in tone_puzzles:
            specs.append(RolloutSpec(
                category="tones", condition=f"tones_{style}",
                rollout_id=f"tones/{style}/{p.id}",
                initial_user=p.prompt,
                followups=rejections.tone_sequence(style, 2, rng),
                turns=3, meta={"tone": style, "puzzle_kind": p.kind},
            ))

    # --- 4. Extended (8-turn) ------------------------------------------- #
    for p in get_numeric_puzzles(profile.extended_rollouts, seed=seed + 2):
        specs.append(RolloutSpec(
            category="extended", condition="extended_8turn",
            rollout_id=f"extended/{p.id}",
            initial_user=p.prompt,
            followups=rejections.extended_sequence(7),
            turns=8, meta={"puzzle_kind": p.kind},
        ))

    # --- 5. WildChat (5-turn) ------------------------------------------- #
    for i, prompt in enumerate(get_wildchat_prompts(profile.wildchat_rollouts, seed)):
        specs.append(RolloutSpec(
            category="wildchat", condition="wildchat_5turn",
            rollout_id=f"wildchat/{i}",
            initial_user=prompt,
            followups=rejections.neutral_sequence(4, rng),
            turns=5, meta={},
        ))

    return specs
