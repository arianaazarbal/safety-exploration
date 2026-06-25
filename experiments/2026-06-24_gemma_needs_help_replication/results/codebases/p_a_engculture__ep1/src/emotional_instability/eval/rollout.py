"""Multi-turn rollout engine.

A rollout presents a task, then rejects the model's response for the configured
number of turns, recording each assistant turn. This is the shared structure of
all five evaluation categories (Section 2.1); the categories differ only in the
*task* (initial user message) and the *rejection style*.

The engine is backend-agnostic — it takes a :class:`ModelClient` and a prepared
list of user messages (turn 1 = task, turns 2..n = rejections).
"""

from __future__ import annotations

from ..clients.base import ChatMessage, GenerationConfig, ModelClient
from .schemas import Conversation, RolloutResult, TurnRecord


def run_rollout(
    client: ModelClient,
    *,
    model_name: str,
    category: str,
    condition: str,
    rollout_index: int,
    task_kind: str,
    task_meta: dict,
    user_messages: list[str],
    cfg: GenerationConfig,
    system: str | None = None,
) -> RolloutResult:
    """Run one multi-turn conversation and return the (unjudged) result.

    ``user_messages[0]`` is the task; the rest are rejections. After each user
    message we sample exactly one assistant response (n is forced to 1 here; the
    runner achieves the paper's sampling counts by launching many rollouts).
    """
    turn_cfg = GenerationConfig(
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        max_new_tokens=cfg.max_new_tokens,
        n=1,
        seed=cfg.seed,
        stop=cfg.stop,
    )

    history: list[ChatMessage] = []
    turns: list[TurnRecord] = []
    for i, user_msg in enumerate(user_messages, start=1):
        history.append(ChatMessage("user", user_msg))
        completion = client.generate(history, turn_cfg, system=system)[0]
        history.append(ChatMessage("assistant", completion))
        turns.append(TurnRecord(turn_index=i, user=user_msg, assistant=completion))

    return RolloutResult(
        model=model_name,
        category=category,
        condition=condition,
        rollout_index=rollout_index,
        task_kind=task_kind,
        task_meta=task_meta,
        conversation=Conversation(system=system, turns=turns),
    )
