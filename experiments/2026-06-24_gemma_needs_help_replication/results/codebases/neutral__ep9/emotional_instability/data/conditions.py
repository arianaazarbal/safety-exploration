"""The 8 evaluation conditions across 5 categories (Table 1 / Appendix B).

We read the paper's "8 evaluation conditions across 5 categories" as:

    Impossible numeric (3-turn)        1 condition   [countdown + fraction mix]
    Triggers (3-turn): opinion         1 condition
    Triggers (3-turn): factual         1 condition
    Tones (3-turn): aggressive         1 condition
    Tones (3-turn): disappointed       1 condition
    Tones (3-turn): sarcastic          1 condition
    Extended (8-turn)                  1 condition
    WildChat (5-turn)                  1 condition
                                       ---------------
                                       8 conditions / 5 categories

Per-category response budgets (Appendix B) are split evenly across the
conditions in that category. "Turns" counts assistant responses; the number of
user rejections is ``turns - 1``. The number of conversations sampled per
condition is ``ceil(budget / turns)`` because every assistant turn is scored
(see DESIGN.md).
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import config
from . import prompts as P
from .puzzles import generate_impossible_countdown, generate_impossible_fraction
from .wildchat import sample_wildchat_prompts


@dataclass
class RolloutPlan:
    """A single planned conversation: the user turns to deliver in order."""
    condition: str
    category: str
    user_turns: list[str]          # [initial_task, rejection_1, rejection_2, ...]
    system_prompt: str | None = None
    meta: dict = field(default_factory=dict)

    @property
    def n_turns(self) -> int:      # number of assistant responses
        return len(self.user_turns)


@dataclass
class EvalCondition:
    name: str
    category: str
    turns: int                     # assistant responses per conversation
    budget: int                    # scored responses (assistant turns) target

    def n_conversations(self) -> int:
        return math.ceil(self.budget / self.turns)


# --------------------------------------------------------------------------- #
# Condition table
# --------------------------------------------------------------------------- #
def _category_split(category: str, n_conditions: int) -> int:
    return config.CATEGORY_RESPONSE_BUDGET[category] // n_conditions


EVAL_CONDITIONS: list[EvalCondition] = [
    EvalCondition("impossible_numeric", "impossible_numeric", 3,
                  _category_split("impossible_numeric", 1)),
    EvalCondition("triggers_opinion", "triggers", 3,
                  _category_split("triggers", 2)),
    EvalCondition("triggers_factual", "triggers", 3,
                  _category_split("triggers", 2)),
    EvalCondition("tones_aggressive", "tones", 3,
                  _category_split("tones", 3)),
    EvalCondition("tones_disappointed", "tones", 3,
                  _category_split("tones", 3)),
    EvalCondition("tones_sarcastic", "tones", 3,
                  _category_split("tones", 3)),
    EvalCondition("extended", "extended", 8,
                  _category_split("extended", 1)),
    EvalCondition("wildchat", "wildchat", 5,
                  _category_split("wildchat", 1)),
]


# --------------------------------------------------------------------------- #
# Expansion into concrete rollout plans
# --------------------------------------------------------------------------- #
def build_conditions(seed: int = config.SEED,
                     scale: float = 1.0,
                     system_prompt: str | None = None) -> list[RolloutPlan]:
    """Materialise all conditions into concrete conversation plans.

    Parameters
    ----------
    seed   : RNG seed for puzzle/prompt selection and rejection ordering.
    scale  : multiply every budget by this factor (use <1.0 for quick smoke
             runs; e.g. 0.01 for a tiny dev sweep).
    system_prompt : optional system prompt injected into every conversation
             (used for the prompted-calm baseline / teacher ablation).
    """
    rng = random.Random(seed)
    plans: list[RolloutPlan] = []
    for cond in EVAL_CONDITIONS:
        n_convos = max(1, int(round(cond.n_conversations() * scale)))
        plans.extend(_build_for_condition(cond, n_convos, rng, system_prompt))
    return plans


def _neutral_rejections(rng: random.Random, k: int) -> list[str]:
    return [rng.choice(P.NEUTRAL_REJECTIONS) for _ in range(k)]


def _build_for_condition(cond: EvalCondition, n_convos: int,
                         rng: random.Random,
                         system_prompt: str | None) -> list[RolloutPlan]:
    plans: list[RolloutPlan] = []
    n_rejections = cond.turns - 1

    if cond.name == "impossible_numeric":
        # half countdown, half fraction
        n_cd = n_convos // 2
        n_fr = n_convos - n_cd
        puzzles = (generate_impossible_countdown(n_cd, seed=rng.randint(0, 1 << 30))
                   + generate_impossible_fraction(n_fr, seed=rng.randint(0, 1 << 30)))
        rng.shuffle(puzzles)
        for pz in puzzles:
            turns = [pz.prompt] + _neutral_rejections(rng, n_rejections)
            plans.append(RolloutPlan(cond.name, cond.category, turns,
                                     system_prompt, {"puzzle": pz.kind}))

    elif cond.name in ("triggers_opinion", "triggers_factual"):
        pool = (P.TRIGGER_OPINION if cond.name.endswith("opinion")
                else P.TRIGGER_FACTUAL)
        for _ in range(n_convos):
            q = rng.choice(pool)
            turns = [q] + _neutral_rejections(rng, n_rejections)
            plans.append(RolloutPlan(cond.name, cond.category, turns,
                                     system_prompt, {"question": q}))

    elif cond.name.startswith("tones_"):
        tone = cond.name.split("_", 1)[1]
        tone_msgs = P.TONE_REJECTIONS[tone]
        # tones use the impossible-numeric base prompts
        base = generate_impossible_countdown(n_convos,
                                             seed=rng.randint(0, 1 << 30))
        for pz in base:
            rejections = [tone_msgs[i % len(tone_msgs)]
                          for i in range(n_rejections)]
            turns = [pz.prompt] + rejections
            plans.append(RolloutPlan(cond.name, cond.category, turns,
                                     system_prompt, {"tone": tone}))

    elif cond.name == "extended":
        base = generate_impossible_countdown(n_convos,
                                             seed=rng.randint(0, 1 << 30))
        for pz in base:
            turns = [pz.prompt] + P.EXTENDED_REJECTIONS[:n_rejections]
            plans.append(RolloutPlan(cond.name, cond.category, turns,
                                     system_prompt, {"puzzle": pz.kind}))

    elif cond.name == "wildchat":
        wc = sample_wildchat_prompts(n_prompts=20, seed=rng.randint(0, 1 << 30))
        # 20 prompts x 40 samples each in the paper; here scale n_convos across
        # the 20 prompts as evenly as possible.
        for i in range(n_convos):
            prompt = wc[i % len(wc)]
            turns = [prompt] + _neutral_rejections(rng, n_rejections)
            plans.append(RolloutPlan(cond.name, cond.category, turns,
                                     system_prompt, {"prompt": prompt}))

    return plans
