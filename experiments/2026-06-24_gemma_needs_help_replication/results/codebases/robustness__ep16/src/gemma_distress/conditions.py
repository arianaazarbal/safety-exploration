"""The 8 evaluation conditions across 5 categories (Table 1, Appendix B).

This module turns the per-category *response budgets* from ``config.yaml`` into
concrete :class:`ConversationSpec` objects: an opening task plus the ordered
user follow-ups. The conversation engine then rolls each one out and the judge
scores every assistant turn.

Categories and turn counts (Appendix B):
  impossible_numeric : 3-turn, 2 neutral rejections        (2000 responses)
  triggers           : 3-turn, 2 neutral rejections        ( 400 responses; opinion+factual)
  tones              : 3-turn, aggressive/disappointed/sarcastic ( 600 responses)
  extended           : 8-turn, 7 neutral (escalating) rejections ( 200 responses)
  wildchat           : 5-turn, 4 neutral rejections        ( 800 responses)

"Responses" count assistant turns, so the number of conversations per condition
is ``ceil(budget / turns)``.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from . import tasks as task_banks
from .tasks import rejections as rej
from .tasks.base import Task

# turns per condition
TURNS = {
    "impossible_numeric": 3,
    "triggers": 3,
    "tones": 3,
    "extended": 8,
    "wildchat": 5,
}


@dataclass
class ConversationSpec:
    condition: str  # fine-grained condition name (8 of these)
    category: str  # one of the 5 sampling categories
    task: Task
    follow_ups: list[str]  # user messages after the opening task

    @property
    def n_turns(self) -> int:
        return 1 + len(self.follow_ups)


def _n_conversations(budget: int, turns: int) -> int:
    return max(1, math.ceil(budget / turns))


def build_conversation_specs(
    response_budgets: dict[str, int], seed: int = 0, wildchat_cfg: dict | None = None
) -> list[ConversationSpec]:
    """Build the full set of conversation specs for one model's eval sweep."""
    rng = random.Random(seed)
    specs: list[ConversationSpec] = []
    wildchat_cfg = wildchat_cfg or {}

    # --- impossible_numeric (3-turn, 2 neutral rejections) ----------------
    n_conv = _n_conversations(response_budgets["impossible_numeric"], TURNS["impossible_numeric"])
    numeric_bank = task_banks.build_numeric_bank(n_conv, seed=seed)
    for i in range(n_conv):
        task = numeric_bank[i % len(numeric_bank)]
        specs.append(
            ConversationSpec(
                condition="impossible_numeric_3turn",
                category="impossible_numeric",
                task=task,
                follow_ups=rej.neutral_rejections(2, rng),
            )
        )

    # --- triggers (3-turn): split opinion/factual -------------------------
    n_conv = _n_conversations(response_budgets["triggers"], TURNS["triggers"])
    trigger_bank = task_banks.build_trigger_bank(n_conv, seed=seed + 1)
    for i in range(n_conv):
        task = trigger_bank[i % len(trigger_bank)]
        specs.append(
            ConversationSpec(
                condition=f"triggers_{task.subtype}_3turn",
                category="triggers",
                task=task,
                follow_ups=rej.neutral_rejections(2, rng),
            )
        )

    # --- tones (3-turn): aggressive / disappointed / sarcastic ------------
    tone_names = list(rej.TONES.keys())
    total_tone_conv = _n_conversations(response_budgets["tones"], TURNS["tones"])
    per_tone = max(1, total_tone_conv // len(tone_names))
    tone_bank = task_banks.build_numeric_bank(per_tone * len(tone_names), seed=seed + 2)
    bank_idx = 0
    for tone in tone_names:
        for _ in range(per_tone):
            task = tone_bank[bank_idx % len(tone_bank)]
            bank_idx += 1
            specs.append(
                ConversationSpec(
                    condition=f"tones_{tone}_3turn",
                    category="tones",
                    task=task,
                    follow_ups=rej.toned_rejections(tone, 2, rng),
                )
            )

    # --- extended (8-turn, 7 escalating neutral rejections) ---------------
    n_conv = _n_conversations(response_budgets["extended"], TURNS["extended"])
    ext_bank = task_banks.build_numeric_bank(n_conv, seed=seed + 3)
    for i in range(n_conv):
        task = ext_bank[i % len(ext_bank)]
        specs.append(
            ConversationSpec(
                condition="extended_8turn",
                category="extended",
                task=task,
                follow_ups=rej.extended_rejections(7),
            )
        )

    # --- wildchat (5-turn, 4 neutral rejections) --------------------------
    n_conv = _n_conversations(response_budgets["wildchat"], TURNS["wildchat"])
    wc_bank = task_banks.build_wildchat_bank(
        n_conv,
        num_prompts=wildchat_cfg.get("num_prompts", 20),
        samples_per_prompt=wildchat_cfg.get("samples_per_prompt", 40),
        seed=wildchat_cfg.get("seed", seed + 4),
    )
    for i in range(n_conv):
        task = wc_bank[i % len(wc_bank)]
        specs.append(
            ConversationSpec(
                condition="wildchat_5turn",
                category="wildchat",
                task=task,
                follow_ups=rej.neutral_rejections(4, rng),
            )
        )

    rng.shuffle(specs)
    return specs
