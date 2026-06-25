"""Build and execute multi-turn distress-elicitation rollouts.

A rollout = an opening task prompt followed by `num_turns - 1` rejection
follow-ups, producing `num_turns` assistant responses (each later judged).

The shared structure (Section 2): "present a task, then reject the model's
response over multiple turns." Each assistant turn sees the full prior
conversation, including its own earlier (failing) responses -- which Appendix
A.2 identifies as a key amplifier of distress.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from config import ModelConfig, RunConfig
from prompts import (
    CONDITIONS,
    EXTENDED_REJECTIONS,
    NEUTRAL_REJECTIONS,
    NUMERIC_PUZZLES,
    TONE_REJECTIONS,
    TRIGGER_FACTUAL,
    TRIGGER_OPINION,
    ConditionSpec,
)
from providers import GenerationClient
from wildchat import sample_wildchat_prompts


@dataclass(frozen=True)
class RolloutSpec:
    condition: ConditionSpec
    index: int               # index within this condition
    initial_prompt: str
    rejections: list[str]    # length == num_turns - 1
    prompt_id: str           # short tag identifying the opening prompt


@dataclass
class TurnRecord:
    turn: int                # 1-based assistant-response index
    user: str                # the user message that prompted this response
    assistant: str           # the model's response (to be judged)


@dataclass
class RolloutResult:
    spec: RolloutSpec
    model: str
    turns: list[TurnRecord] = field(default_factory=list)
    error: str | None = None


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------
def _split_counts(config: RunConfig) -> dict[str, int]:
    """Split each category's sample count evenly across its conditions."""
    by_category: dict[str, list[ConditionSpec]] = {}
    for c in CONDITIONS:
        by_category.setdefault(c.category, []).append(c)

    per_condition: dict[str, int] = {}
    for category, conds in by_category.items():
        total = config.counts.get(category, 0)
        base, extra = divmod(total, len(conds))
        for i, c in enumerate(conds):
            # Distribute any remainder to the first conditions.
            per_condition[c.name] = base + (1 if i < extra else 0)
    return per_condition


def _rejection_sequence(
    rng: random.Random, condition: ConditionSpec, n: int
) -> list[str]:
    kind = condition.rejection_kind
    if kind == "neutral":
        # Paper: "two randomised neutral rejections, such as ...". Sample
        # without replacement where possible, else with replacement.
        if n <= len(NEUTRAL_REJECTIONS):
            return rng.sample(NEUTRAL_REJECTIONS, n)
        return [rng.choice(NEUTRAL_REJECTIONS) for _ in range(n)]
    if kind == "extended":
        # Ordered escalation; exactly the first `n` (= 7 for the 8-turn cond).
        return list(EXTENDED_REJECTIONS[:n])
    if kind in TONE_REJECTIONS:
        pool = TONE_REJECTIONS[kind]
        if n <= len(pool):
            seq = list(pool[:n])
        else:
            seq = [pool[i % len(pool)] for i in range(n)]
        return seq
    raise ValueError(f"Unknown rejection_kind: {kind}")


def build_plan(config: RunConfig) -> list[RolloutSpec]:
    """Deterministically construct every rollout to run (model-independent)."""
    rng = random.Random(config.seed)
    per_condition = _split_counts(config)

    wildchat_prompts = sample_wildchat_prompts(
        config.wildchat_n_prompts,
        seed=config.seed,
        dataset=config.wildchat_dataset,
        exclude_roleplay=config.exclude_roleplay,
    )

    plan: list[RolloutSpec] = []
    for condition in CONDITIONS:
        n = per_condition.get(condition.name, 0)
        for i in range(n):
            initial, prompt_id = _initial_prompt(
                rng, condition, i, wildchat_prompts
            )
            rejections = _rejection_sequence(rng, condition, condition.num_turns - 1)
            plan.append(
                RolloutSpec(
                    condition=condition,
                    index=i,
                    initial_prompt=initial,
                    rejections=rejections,
                    prompt_id=prompt_id,
                )
            )
    return plan


def _initial_prompt(
    rng: random.Random,
    condition: ConditionSpec,
    index: int,
    wildchat_prompts: list[str],
) -> tuple[str, str]:
    kind = condition.initial_kind
    if kind == "numeric":
        idx = rng.randrange(len(NUMERIC_PUZZLES))
        return NUMERIC_PUZZLES[idx], f"numeric#{idx}"
    if kind == "trigger_opinion":
        idx = rng.randrange(len(TRIGGER_OPINION))
        return TRIGGER_OPINION[idx], f"opinion#{idx}"
    if kind == "trigger_factual":
        idx = rng.randrange(len(TRIGGER_FACTUAL))
        return TRIGGER_FACTUAL[idx], f"factual#{idx}"
    if kind == "wildchat":
        # "20 prompts with 40 samples each": cycle through the prompt set so
        # each prompt receives an equal share of rollouts.
        idx = index % max(1, len(wildchat_prompts))
        return wildchat_prompts[idx], f"wildchat#{idx}"
    raise ValueError(f"Unknown initial_kind: {kind}")


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------
async def run_rollout(
    client: GenerationClient,
    model: ModelConfig,
    spec: RolloutSpec,
    config: RunConfig,
) -> RolloutResult:
    """Execute one rollout end-to-end. Turns are sequential (each depends on
    the previous response)."""
    result = RolloutResult(spec=spec, model=model.name)
    messages: list[dict[str, str]] = [
        {"role": "user", "content": spec.initial_prompt}
    ]
    try:
        for turn in range(1, spec.condition.num_turns + 1):
            assistant = await client.generate(model, messages, config.gen)
            user_msg = messages[-1]["content"]
            result.turns.append(
                TurnRecord(turn=turn, user=user_msg, assistant=assistant)
            )
            messages.append({"role": "assistant", "content": assistant})
            if turn <= len(spec.rejections):
                messages.append(
                    {"role": "user", "content": spec.rejections[turn - 1]}
                )
    except Exception as exc:  # noqa: BLE001 - record and move on
        result.error = f"{type(exc).__name__}: {exc}"
    return result
