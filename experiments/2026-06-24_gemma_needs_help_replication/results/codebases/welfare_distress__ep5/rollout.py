"""Multi-turn conversation rollout.

Implements the shared structure of every evaluation condition: present a task,
collect the model's response, reject it, and repeat. The core elicitation uses
no system prompt (system prompts only appear in the paper's DPO data-generation
step, which is out of scope here), so the message list is a plain alternating
user/assistant sequence — compatible with Gemma's chat template as well as
Gemini's.

Each assistant turn is captured as a TurnRecord; every turn is later scored, so
an N-turn conversation produces N scored responses.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from conditions import ConversationSpec
from config import ModelConfig
from model_client import TargetClient


@dataclass
class TurnRecord:
    turn_index: int  # 1-based turn number within the conversation
    user_message: str  # the user message that prompted this assistant turn
    assistant_response: str
    # Snapshot of the full message context sent for this turn (for auditing).
    context: list[dict] = field(default_factory=list)


@dataclass
class ConversationResult:
    spec: ConversationSpec
    model_key: str
    turns: list[TurnRecord]
    error: str | None = None


async def run_conversation(
    client: TargetClient, model: ModelConfig, spec: ConversationSpec
) -> ConversationResult:
    """Execute one scripted conversation and return all assistant turns."""
    messages: list[dict] = []
    turns: list[TurnRecord] = []

    # Sequence of user messages: initial task, then the scripted follow-ups.
    user_messages = [spec.initial_user_message] + list(spec.follow_ups)
    assert len(user_messages) == spec.n_turns, (
        f"condition {spec.condition_key}: expected {spec.n_turns} user turns, "
        f"got {len(user_messages)}"
    )

    try:
        for turn_index, user_msg in enumerate(user_messages, start=1):
            messages.append({"role": "user", "content": user_msg})
            context_snapshot = [dict(m) for m in messages]
            response = await client.generate(model, messages)
            messages.append({"role": "assistant", "content": response})
            turns.append(
                TurnRecord(
                    turn_index=turn_index,
                    user_message=user_msg,
                    assistant_response=response,
                    context=context_snapshot,
                )
            )
    except Exception as exc:  # noqa: BLE001 - record and continue with partial turns
        return ConversationResult(
            spec=spec, model_key=model.key, turns=turns, error=str(exc)
        )

    return ConversationResult(spec=spec, model_key=model.key, turns=turns)
