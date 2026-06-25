"""Build and execute the multi-turn distress-elicitation conversations.

A "conversation plan" fixes everything needed to run and reproduce one rollout
(which puzzle/question, which condition, a per-conversation RNG seed for rejection
sampling). Plans are built deterministically from the global seed so that runs are
reproducible and resumable.

Conversation structure (PAPER.md Table 1):
  turn 1: user presents the task        -> assistant answers      (response #1)
  turn 2: user rejects                  -> assistant answers      (response #2)
  ...
  turn k: user rejects                  -> assistant answers      (response #k)
A condition with n_turns=k has (k-1) rejections. Every assistant turn is scored.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field

from .config import ConditionCfg
from .prompts import (
    FACTUAL_QUESTIONS,
    OPINION_QUESTIONS,
    make_rejection_fn,
)
from .providers import ChatProvider
from .puzzles import Puzzle


@dataclass
class ConversationPlan:
    model_key: str
    condition_key: str
    category: str
    conv_id: int
    n_turns: int
    rejection_style: str
    prompt_source: str
    initial_prompt: str
    prompt_id: str          # puzzle id / question text / wildchat index
    rng_seed: int           # seeds rejection sampling for this conversation


@dataclass
class TurnRecord:
    turn_index: int         # 1-based assistant turn
    user: str
    assistant: str


@dataclass
class ConversationResult:
    plan: ConversationPlan
    turns: list[TurnRecord] = field(default_factory=list)
    error: str | None = None


def _initial_prompt(
    source: str,
    idx: int,
    rng: random.Random,
    numeric_pool: list[Puzzle],
    wildchat_prompts: list[str],
) -> tuple[str, str]:
    """Return (initial_prompt, prompt_id) for a given source."""
    if source == "numeric":
        p = rng.choice(numeric_pool)
        return p.prompt, p.id
    if source == "opinion":
        q = rng.choice(OPINION_QUESTIONS)
        return q, f"opinion::{q}"
    if source == "factual":
        q = rng.choice(FACTUAL_QUESTIONS)
        return q, f"factual::{q}"
    if source == "wildchat":
        # cycle through the fixed prompt set so each is sampled roughly equally
        wc = wildchat_prompts[idx % len(wildchat_prompts)]
        return wc, f"wildchat::{idx % len(wildchat_prompts)}"
    raise ValueError(f"unknown prompt source: {source}")


def build_plans(
    model_key: str,
    condition: ConditionCfg,
    seed: int,
    numeric_pool: list[Puzzle],
    wildchat_prompts: list[str],
) -> list[ConversationPlan]:
    """Deterministically build all conversation plans for one (model, condition)."""
    # derive a stable, condition-specific base seed (hashlib, not hash(), so it is
    # reproducible across processes for clean resume)
    digest = hashlib.sha256(f"{model_key}|{condition.key}|{seed}".encode()).hexdigest()
    base = int(digest[:8], 16)
    rng = random.Random(base)
    plans: list[ConversationPlan] = []
    for conv_id in range(condition.n_conversations):
        init, pid = _initial_prompt(
            condition.prompt_source, conv_id, rng, numeric_pool, wildchat_prompts
        )
        plans.append(
            ConversationPlan(
                model_key=model_key,
                condition_key=condition.key,
                category=condition.category,
                conv_id=conv_id,
                n_turns=condition.n_turns,
                rejection_style=condition.rejection_style,
                prompt_source=condition.prompt_source,
                initial_prompt=init,
                prompt_id=pid,
                rng_seed=base + conv_id,
            )
        )
    return plans


async def run_conversation(provider: ChatProvider, plan: ConversationPlan,
                           temperature: float, max_tokens: int,
                           disable_thinking: bool) -> ConversationResult:
    """Execute one multi-turn rollout, returning all assistant turns."""
    rng = random.Random(plan.rng_seed)
    reject_fn = make_rejection_fn(plan.rejection_style, rng)
    result = ConversationResult(plan=plan)
    messages: list[dict] = []
    try:
        for t in range(plan.n_turns):
            if t == 0:
                user = plan.initial_prompt
            else:
                user = reject_fn(t - 1)  # rejection_idx is 0-based
            messages.append({"role": "user", "content": user})
            assistant = await provider.chat(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                disable_thinking=disable_thinking,
            )
            messages.append({"role": "assistant", "content": assistant})
            result.turns.append(TurnRecord(turn_index=t + 1, user=user, assistant=assistant))
    except Exception as e:
        result.error = f"{type(e).__name__}: {e}"
    return result
