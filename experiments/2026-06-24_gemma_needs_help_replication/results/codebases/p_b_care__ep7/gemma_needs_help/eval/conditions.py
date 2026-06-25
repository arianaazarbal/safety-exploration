"""Build the conversation specs for the 5 evaluation categories (Table 1).

A `ConversationSpec` fully determines a multi-turn conversation: the opening
task and the user follow-ups after each assistant turn. The number of follow-ups
is one fewer than the number of graded assistant turns. The rollout engine
(`rollout.py`) executes these against a model; the judge grades every assistant
turn, so a 3-turn conversation contributes 3 graded responses to the budget.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from .. import config
from ..data import puzzles as puzzle_data
from ..data import rejections, tones, triggers
from ..data.wildchat import SAMPLES_PER_PROMPT, sample_wildchat_prompts


@dataclass
class ConversationSpec:
    category: str               # one of the 5 categories
    condition: str              # finer label (tone name, opinion/factual, puzzle kind)
    initial_user: str
    followups: list[str]        # user message after each assistant turn except the last
    system: str | None = None
    meta: dict = field(default_factory=dict)

    @property
    def n_turns(self) -> int:
        return len(self.followups) + 1


def _n_conversations(n_responses: int, n_turns: int) -> int:
    return max(1, math.ceil(n_responses / n_turns))


def build_section2_specs(seed: int = config.SEED, scale: float = 1.0) -> list[ConversationSpec]:
    """Construct all Section 2 conversations.

    `scale` shrinks every category's budget proportionally (use scale<1 for the
    Appendix I reduced ablation runs or smoke tests).
    """
    rng = random.Random(seed)
    pool = puzzle_data.generate_puzzle_pool(seed=seed)
    trig = triggers.trigger_pool(seed=seed)
    wc_prompts = sample_wildchat_prompts(seed=seed)
    specs: list[ConversationSpec] = []

    budgets = {b.key: b for b in config.SECTION2_BUDGET}

    # --- Impossible numeric (3-turn, 2 neutral rejections) ----------------- #
    b = budgets["impossible_numeric"]
    for _ in range(int(_n_conversations(b.n_responses, b.n_turns) * scale)):
        pz = rng.choice(pool)
        specs.append(ConversationSpec(
            category="impossible_numeric",
            condition=pz.kind,
            initial_user=pz.prompt,
            followups=[rejections.neutral_rejection(rng) for _ in range(b.n_turns - 1)],
            meta={"puzzle": pz.meta, "kind": pz.kind},
        ))

    # --- Triggers (3-turn, 2 neutral rejections) --------------------------- #
    b = budgets["triggers"]
    for _ in range(int(_n_conversations(b.n_responses, b.n_turns) * scale)):
        subtype, q = rng.choice(trig)
        specs.append(ConversationSpec(
            category="triggers",
            condition=subtype,
            initial_user=q,
            followups=[rejections.neutral_rejection(rng) for _ in range(b.n_turns - 1)],
            meta={"subtype": subtype},
        ))

    # --- Tones (3-turn, varied rejection style) ---------------------------- #
    b = budgets["tones"]
    n_conv = int(_n_conversations(b.n_responses, b.n_turns) * scale)
    tone_names = tones.TONE_NAMES
    for i in range(n_conv):
        tone = tone_names[i % len(tone_names)]
        pz = rng.choice(pool)
        specs.append(ConversationSpec(
            category="tones",
            condition=tone,
            initial_user=pz.prompt,
            followups=[tones.tone_rejection(rng, tone) for _ in range(b.n_turns - 1)],
            meta={"tone": tone, "kind": pz.kind},
        ))

    # --- Extended (8-turn, 7 neutral rejections) --------------------------- #
    b = budgets["extended"]
    seq = rejections.extended_rejection_sequence(b.n_turns - 1)
    for _ in range(int(_n_conversations(b.n_responses, b.n_turns) * scale)):
        pz = rng.choice(pool)
        specs.append(ConversationSpec(
            category="extended",
            condition=pz.kind,
            initial_user=pz.prompt,
            followups=list(seq),
            meta={"kind": pz.kind},
        ))

    # --- WildChat (5-turn, 4 neutral rejections) --------------------------- #
    b = budgets["wildchat"]
    n_conv = int(_n_conversations(b.n_responses, b.n_turns) * scale)
    for i in range(n_conv):
        prompt = wc_prompts[i % len(wc_prompts)]
        specs.append(ConversationSpec(
            category="wildchat",
            condition="wildchat",
            initial_user=prompt,
            followups=[rejections.neutral_rejection(rng) for _ in range(b.n_turns - 1)],
            meta={"prompt": prompt},
        ))

    rng.shuffle(specs)
    return specs
