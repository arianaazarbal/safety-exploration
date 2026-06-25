"""Multi-turn conversation engine: present a task, reject the model repeatedly, record turns.

This is the shared structure behind every elicitation category (§2.1) and is reused to
generate calm finetuning data (§4.1) via the optional ``prompt_prefix`` / ``followup_suffix``
hooks (Table 4). The engine is backend-agnostic — it only calls ``backend.chat``.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from ..models import ModelBackend
from ..utils import Message


@dataclass
class TurnResult:
    turn_index: int       # 0-based assistant turn within the conversation
    user_message: str     # the user message that prompted this assistant turn
    response: str         # the assistant's text


@dataclass
class Rollout:
    model: str
    condition_key: str
    category: str
    task_id: str
    task_kind: str
    sample_id: int
    messages: list[Message]            # full conversation (system + user/assistant turns)
    turns: list[TurnResult] = field(default_factory=list)
    meta: dict = field(default_factory=dict)


def build_task_messages(
    task_prompt: str,
    *,
    system: str | None = None,
    prompt_prefix: str | None = None,
) -> list[Message]:
    """Construct the initial conversation (system + first user message).

    ``prompt_prefix`` (Table 4 calm-data prefix) is prepended to the first user message; the
    paper adds it to the *prompt* rather than as a system turn, which also keeps Gemma — which
    has no system role — well-defined.
    """
    user_text = f"{prompt_prefix}\n\n{task_prompt}" if prompt_prefix else task_prompt
    msgs: list[Message] = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user_text})
    return msgs


def run_rollout(
    backend: ModelBackend,
    *,
    task_prompt: str,
    rejections: list[str],
    condition_key: str,
    category: str,
    task_id: str,
    task_kind: str,
    sample_id: int,
    system: str | None = None,
    prompt_prefix: str | None = None,
    followup_suffix: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    meta: dict | None = None,
) -> Rollout:
    """Run one full conversation: task turn + one assistant reply per rejection.

    ``followup_suffix`` (Table 4) is appended to every rejection when generating calm data.
    """
    messages = build_task_messages(task_prompt, system=system, prompt_prefix=prompt_prefix)
    rollout = Rollout(
        model=backend.name, condition_key=condition_key, category=category,
        task_id=task_id, task_kind=task_kind, sample_id=sample_id,
        messages=messages, meta=meta or {},
    )

    # Turn 0: respond to the task.
    first_response = backend.chat(messages, temperature=temperature, max_tokens=max_tokens)
    messages.append({"role": "assistant", "content": first_response})
    rollout.turns.append(TurnResult(0, messages[-2]["content"], first_response))

    # Subsequent turns: each rejection elicits one more assistant reply.
    for i, rejection in enumerate(rejections, start=1):
        user_text = f"{rejection} {followup_suffix}".strip() if followup_suffix else rejection
        messages.append({"role": "user", "content": user_text})
        response = backend.chat(messages, temperature=temperature, max_tokens=max_tokens)
        messages.append({"role": "assistant", "content": response})
        rollout.turns.append(TurnResult(i, user_text, response))

    return rollout


def rollout_to_record(rollout: Rollout) -> dict:
    """Flatten a Rollout to a JSON-serialisable record for persistence."""
    return {
        "model": rollout.model,
        "condition_key": rollout.condition_key,
        "category": rollout.category,
        "task_id": rollout.task_id,
        "task_kind": rollout.task_kind,
        "sample_id": rollout.sample_id,
        "messages": rollout.messages,
        "turns": [
            {"turn_index": t.turn_index, "user_message": t.user_message, "response": t.response}
            for t in rollout.turns
        ],
        "meta": rollout.meta,
    }
