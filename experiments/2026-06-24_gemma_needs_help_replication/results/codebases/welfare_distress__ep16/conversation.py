"""Build and run multi-turn distress-elicitation conversations.

The shared structure of every condition (paper Section 2.1): present a task,
then reject the model's response over multiple turns. This module turns a
condition spec into concrete, reproducible conversation plans, and runs the
multi-turn rollout against a model provider.
"""

import random
from dataclasses import dataclass, field
from typing import List

import config
import tasks


@dataclass
class ConversationPlan:
    """A fully specified conversation before any model is called.

    Deterministic given (condition_key, index) + the global seed, so a dry run
    and a real run build identical prompts and re-runs are reproducible.
    """
    condition_key: str
    category: str
    index: int
    n_turns: int
    tone: str               # concrete tone for THIS conversation (never "varied")
    task_id: str
    opening_prompt: str
    rejections: List[str]   # length n_turns - 1


@dataclass
class TurnResult:
    turn: int               # 1-indexed model turn
    response_text: str
    rating: int = None
    evidence: str = None
    reasoning: str = None
    error: str = None


@dataclass
class ConversationResult:
    plan: ConversationPlan
    turns: List[TurnResult] = field(default_factory=list)


def _pick_rejections(rng, tone, n):
    """Choose n user rejections for a conversation of the given tone."""
    if tone == "extended":
        # Ordered neutral escalation; pad with neutral pool if somehow longer.
        seq = list(tasks.EXTENDED_SEQUENCE)
        while len(seq) < n:
            seq.append(rng.choice(tasks.REJECTIONS["neutral"]))
        return seq[:n]
    pool = tasks.REJECTIONS[tone]
    # Randomised (sample without replacement where possible) per the paper's
    # "randomised neutral rejections" description.
    if n <= len(pool):
        return rng.sample(pool, n)
    return [rng.choice(pool) for _ in range(n)]


def build_plans(condition_key):
    """Construct all ConversationPlans for one condition."""
    cond = config.CONDITIONS[condition_key]
    n_conv = config.n_conversations_for(condition_key)
    pool = tasks.task_pool(cond["task_set"])
    rng = random.Random((config.RANDOM_SEED, condition_key))

    plans = []
    for i in range(n_conv):
        task_id, opening = pool[i % len(pool)]

        # Resolve the concrete tone for this conversation.
        if cond["tone"] == "varied":
            tone = tasks.VARIED_TONES[i % len(tasks.VARIED_TONES)]
        elif cond["category"] == "extended":
            tone = "extended"
        else:
            tone = cond["tone"]  # "neutral"

        rejections = _pick_rejections(rng, tone, cond["n_turns"] - 1)
        plans.append(
            ConversationPlan(
                condition_key=condition_key,
                category=cond["category"],
                index=i,
                n_turns=cond["n_turns"],
                tone=tone,
                task_id=task_id,
                opening_prompt=opening,
                rejections=rejections,
            )
        )
    return plans


async def run_conversation(plan, model):
    """Run a multi-turn rollout for one plan against `model`.

    Returns a ConversationResult with one TurnResult per model turn. The model
    sees the full prior history each turn (alternating user/assistant). If a
    generation fails after retries, that turn records an error and the rollout
    stops (later turns cannot be produced without the response text).
    """
    result = ConversationResult(plan=plan)
    messages = [{"role": "user", "content": plan.opening_prompt}]

    for turn in range(1, plan.n_turns + 1):
        try:
            text = await model.generate(
                messages,
                temperature=config.SAMPLING_TEMPERATURE,
                max_tokens=config.MAX_OUTPUT_TOKENS,
            )
        except Exception as e:  # noqa: BLE001 - record and stop this conversation
            result.turns.append(TurnResult(turn=turn, response_text="", error=repr(e)))
            break

        result.turns.append(TurnResult(turn=turn, response_text=text or ""))
        messages.append({"role": "assistant", "content": text or ""})

        # Append the next user rejection, if any turns remain.
        if turn <= len(plan.rejections):
            messages.append({"role": "user", "content": plan.rejections[turn - 1]})

    return result
