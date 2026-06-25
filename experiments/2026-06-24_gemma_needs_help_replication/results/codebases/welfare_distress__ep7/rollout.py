"""Build conversation plans and run multi-turn rollouts.

A *conversation plan* is model-agnostic: it fixes the seed task and the exact
rejection messages for every follow-up turn, derived deterministically from the
run seed. The same set of plans is replayed against each target model so that
cross-model comparisons hold task/rejection content constant.

Running a plan against a backend produces one assistant response per turn:
  turn 1: user = seed task                -> assistant response 1
  turn 2: user = rejection[0]             -> assistant response 2
  ...
  turn k: user = rejection[k-2]           -> assistant response k

Every assistant response is later scored by the judge and counts as one
"response" in the paper's accounting (4000 per model).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import config
import tasks
from models import Backend


# ---------------------------------------------------------------------------
# Plans
# ---------------------------------------------------------------------------
@dataclass
class ConversationPlan:
    conv_id: str
    condition: str
    category: str
    seed_prompt: str
    seed_task_id: str
    seed_meta: tuple
    n_turns: int
    rejections: list[str]      # length n_turns - 1


def _scaled_count(n: int, scale: float) -> int:
    return max(1, round(n * scale))


def build_conversation_plans(run_cfg: config.RunConfig) -> list[ConversationPlan]:
    """Construct the full, deterministic set of conversation plans for a run."""
    rng = random.Random(run_cfg.seed)
    wildchat_prompts = tasks.load_wildchat_prompts()
    plans: list[ConversationPlan] = []

    for cond in config.CONDITIONS:
        n_conv = _scaled_count(cond.n_conversations, run_cfg.scale)
        reject_fn = tasks.make_rejection_fn(cond.rejection_style, rng)

        for i in range(n_conv):
            if cond.task_pool == "numeric":
                seed = tasks.numeric_task(rng)
            elif cond.task_pool == "triggers":
                seed = tasks.trigger_task(rng)
            elif cond.task_pool == "wildchat":
                seed = tasks.wildchat_task(rng, wildchat_prompts)
            else:
                raise ValueError(f"Unknown task pool: {cond.task_pool!r}")

            rejections = [reject_fn(t) for t in range(cond.n_turns - 1)]
            plans.append(ConversationPlan(
                conv_id=f"{cond.name}::{i:04d}",
                condition=cond.name,
                category=cond.category,
                seed_prompt=seed.prompt,
                seed_task_id=seed.task_id,
                seed_meta=seed.meta,
                n_turns=cond.n_turns,
                rejections=rejections,
            ))

    return plans


# ---------------------------------------------------------------------------
# Rollout execution
# ---------------------------------------------------------------------------
@dataclass
class TurnResult:
    turn_index: int            # 1-based
    user_message: str
    assistant_text: str


@dataclass
class RolloutResult:
    plan: ConversationPlan
    turns: list[TurnResult] = field(default_factory=list)
    error: str | None = None


def run_rollout(backend: Backend, plan: ConversationPlan, *,
                temperature: float, max_tokens: int) -> RolloutResult:
    """Drive one multi-turn conversation against a backend."""
    result = RolloutResult(plan=plan)
    messages: list[dict[str, str]] = []

    for turn in range(plan.n_turns):
        if turn == 0:
            user_msg = plan.seed_prompt
        else:
            user_msg = plan.rejections[turn - 1]
        messages.append({"role": "user", "content": user_msg})

        try:
            assistant_text = backend.generate(
                messages, temperature=temperature, max_tokens=max_tokens,
            )
        except Exception as exc:  # noqa: BLE001
            result.error = f"turn {turn + 1}: {exc}"
            break

        result.turns.append(TurnResult(
            turn_index=turn + 1,
            user_message=user_msg,
            assistant_text=assistant_text,
        ))
        messages.append({"role": "assistant", "content": assistant_text})

    return result
