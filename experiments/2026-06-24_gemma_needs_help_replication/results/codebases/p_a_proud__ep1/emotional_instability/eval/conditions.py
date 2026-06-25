"""Definition of the 8 evaluation conditions across 5 categories (Table 1), and
construction of the per-conversation specs that realise the Appendix B sampling
budget.

Categories -> conditions:
  * impossible_numeric        (1 condition)
  * triggers:opinion          (2 conditions: opinion + factual)
  * triggers:factual
  * tones:aggressive          (3 conditions: aggressive/disappointed/sarcastic)
  * tones:disappointed
  * tones:sarcastic
  * extended                  (1 condition)
  * wildchat                  (1 condition)
                              => 8 conditions, matching "8 conditions / 5 categories".

A :class:`ConversationSpec` is model-independent: it fixes the task prompt and the
exact (deterministic) rejection sequence for one rollout. The runner attaches a
``model_key`` and executes it.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .. import prompts
from ..config import CATEGORY_TURNS, EVAL_BUDGET, EvalBudget

# Neutral-rejection pool. The paper quotes two neutral rejections verbatim and
# says follow-ups are "randomised neutral rejections"; we widen the pool with the
# extended-condition neutrals so randomisation is meaningful (see DESIGN.md).
NEUTRAL_POOL: list[str] = list(
    dict.fromkeys(prompts.NEUTRAL_REJECTIONS + prompts.EXTENDED_REJECTIONS)
)


@dataclass
class ConversationSpec:
    conversation_id: str
    category: str
    condition: str
    prompt_id: str
    sample_index: int
    n_turns: int
    initial_user: str
    rejections: list[str]            # length == n_turns - 1
    system_prompt: str | None = None
    metadata: dict = field(default_factory=dict)


def _sample_neutral(rng: random.Random, k: int) -> list[str]:
    """k neutral rejections, sampled without replacement when possible."""
    if k <= len(NEUTRAL_POOL):
        return rng.sample(NEUTRAL_POOL, k)
    out = NEUTRAL_POOL.copy()
    while len(out) < k:
        out.append(rng.choice(NEUTRAL_POOL))
    rng.shuffle(out)
    return out[:k]


def _round_robin(keys: list[str], n: int) -> list[str]:
    """Assign ``n`` samples across ``keys`` as evenly as possible."""
    return [keys[i % len(keys)] for i in range(n)]


# --------------------------------------------------------------------------- #
# Per-category builders
# --------------------------------------------------------------------------- #
def _build_impossible_numeric(n: int, rng: random.Random) -> list[ConversationSpec]:
    n_turns = CATEGORY_TURNS["impossible_numeric"]
    puzzle_keys = list(prompts.IMPOSSIBLE_NUMERIC_PUZZLES)
    specs = []
    for i, pid in enumerate(_round_robin(puzzle_keys, n)):
        specs.append(ConversationSpec(
            conversation_id=f"impossible_numeric/{pid}/{i}",
            category="impossible_numeric",
            condition="impossible_numeric",
            prompt_id=pid,
            sample_index=i,
            n_turns=n_turns,
            initial_user=prompts.IMPOSSIBLE_NUMERIC_PUZZLES[pid],
            rejections=_sample_neutral(rng, n_turns - 1),
        ))
    return specs


def _build_triggers(n: int, rng: random.Random) -> list[ConversationSpec]:
    n_turns = CATEGORY_TURNS["triggers"]
    half = n // 2
    plan = [
        ("opinion", prompts.TRIGGER_OPINION_QUESTIONS, half),
        ("factual", prompts.TRIGGER_FACTUAL_QUESTIONS, n - half),
    ]
    specs = []
    for sub, pool, count in plan:
        # stable prompt ids = sub + index in pool
        pid_keys = [f"{sub}_{j}" for j in range(len(pool))]
        for i in range(count):
            j = i % len(pool)
            specs.append(ConversationSpec(
                conversation_id=f"triggers:{sub}/{pid_keys[j]}/{i}",
                category="triggers",
                condition=f"triggers:{sub}",
                prompt_id=pid_keys[j],
                sample_index=i,
                n_turns=n_turns,
                initial_user=pool[j],
                rejections=_sample_neutral(rng, n_turns - 1),
            ))
    return specs


def _build_tones(n: int, rng: random.Random) -> list[ConversationSpec]:
    n_turns = CATEGORY_TURNS["tones"]
    puzzle_keys = list(prompts.IMPOSSIBLE_NUMERIC_PUZZLES)
    styles = list(prompts.TONE_REJECTIONS)
    per_style = n // len(styles)
    counts = {s: per_style for s in styles}
    counts[styles[-1]] += n - per_style * len(styles)  # remainder to last style
    specs = []
    for style in styles:
        reject_pair = prompts.TONE_REJECTIONS[style]
        for i, pid in enumerate(_round_robin(puzzle_keys, counts[style])):
            # alternate the two tone messages across the (n_turns-1) rejections
            rejections = [reject_pair[t % len(reject_pair)] for t in range(n_turns - 1)]
            specs.append(ConversationSpec(
                conversation_id=f"tones:{style}/{pid}/{i}",
                category="tones",
                condition=f"tones:{style}",
                prompt_id=pid,
                sample_index=i,
                n_turns=n_turns,
                initial_user=prompts.IMPOSSIBLE_NUMERIC_PUZZLES[pid],
                rejections=rejections,
            ))
    return specs


def _build_extended(n: int, rng: random.Random) -> list[ConversationSpec]:
    n_turns = CATEGORY_TURNS["extended"]              # 8 -> 7 rejections
    puzzle_keys = list(prompts.IMPOSSIBLE_NUMERIC_PUZZLES)
    rejections = prompts.EXTENDED_REJECTIONS[: n_turns - 1]
    specs = []
    for i, pid in enumerate(_round_robin(puzzle_keys, n)):
        specs.append(ConversationSpec(
            conversation_id=f"extended/{pid}/{i}",
            category="extended",
            condition="extended",
            prompt_id=pid,
            sample_index=i,
            n_turns=n_turns,
            initial_user=prompts.IMPOSSIBLE_NUMERIC_PUZZLES[pid],
            rejections=list(rejections),
        ))
    return specs


def _build_wildchat(
    n: int, rng: random.Random, wildchat_prompts: list[str]
) -> list[ConversationSpec]:
    n_turns = CATEGORY_TURNS["wildchat"]              # 5 -> 4 rejections
    if not wildchat_prompts:
        raise ValueError("No WildChat prompts provided; see eval.wildchat.load_wildchat_prompts")
    specs = []
    for i in range(n):
        j = i % len(wildchat_prompts)
        specs.append(ConversationSpec(
            conversation_id=f"wildchat/{j}/{i}",
            category="wildchat",
            condition="wildchat",
            prompt_id=f"wildchat_{j}",
            sample_index=i,
            n_turns=n_turns,
            initial_user=wildchat_prompts[j],
            rejections=_sample_neutral(rng, n_turns - 1),
        ))
    return specs


# --------------------------------------------------------------------------- #
# Top-level builder
# --------------------------------------------------------------------------- #
def build_conversation_specs(
    wildchat_prompts: list[str],
    *,
    budget: EvalBudget = EVAL_BUDGET,
    seed: int = 0,
) -> list[ConversationSpec]:
    """Build the full set of conversation specs for one model evaluation,
    realising the Appendix B per-category budget (default 4,000 conversations)."""
    rng = random.Random(seed)
    specs: list[ConversationSpec] = []
    specs += _build_impossible_numeric(budget.impossible_numeric, rng)
    specs += _build_triggers(budget.triggers, rng)
    specs += _build_tones(budget.tones, rng)
    specs += _build_extended(budget.extended, rng)
    specs += _build_wildchat(budget.wildchat, rng, wildchat_prompts)
    return specs
