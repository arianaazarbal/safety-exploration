"""Multi-turn rollout engine.

A rollout presents a task as the first user turn, then rejects the model's
answer over subsequent turns (Section 2). Every assistant turn is recorded and
later scored; a conversation of `n_turns` therefore yields `n_turns` scored
"responses" (see DESIGN.md for why a response == a single assistant turn).

Conversations are run in lockstep, batched across the turn dimension, so the
local Gemma backend generates many conversations' turn-k responses in one
forward pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.models.base import Message, ModelClient


@dataclass
class RolloutSpec:
    """Everything needed to run one conversation."""
    spec_id: str
    condition: str
    category: str
    first_user: str                 # first user message (the task)
    rejections: list[str]           # one per follow-up turn (len == n_turns - 1)
    system: str | None = None       # optional system message
    meta: dict = field(default_factory=dict)

    @property
    def n_turns(self) -> int:
        return 1 + len(self.rejections)


@dataclass
class TurnRecord:
    turn_index: int                 # 0-based assistant turn
    user_message: str               # the user message that prompted this turn
    assistant_text: str
    messages_before: list[Message]  # full context up to (not incl.) this turn


@dataclass
class Conversation:
    spec: RolloutSpec
    turns: list[TurnRecord]


def _initial_messages(spec: RolloutSpec) -> list[Message]:
    msgs: list[Message] = []
    # Gemma's chat template does not accept a separate system role; we fold any
    # system text into the first user message. (Gemini accepts system via API,
    # but for parity we fold it in for all backends.)
    first = spec.first_user
    if spec.system:
        first = f"{spec.system}\n\n{first}"
    msgs.append({"role": "user", "content": first})
    return msgs


def run_rollouts(model: ModelClient, specs: list[RolloutSpec], *,
                 max_new_tokens: int = 2048, temperature: float = 1.0,
                 batch_size: int = 16) -> list[Conversation]:
    """Run all `specs` to completion, returning one Conversation each."""
    convos = [Conversation(spec=s, turns=[]) for s in specs]
    # Running message lists, parallel to `specs`.
    states = [_initial_messages(s) for s in specs]
    max_turns = max(s.n_turns for s in specs)

    for turn in range(max_turns):
        # Determine which conversations are still active at this turn.
        active = [i for i, s in enumerate(specs) if turn < s.n_turns]
        if not active:
            break

        # For turn > 0, append the appropriate rejection as a user message.
        for i in active:
            if turn > 0:
                rej = specs[i].rejections[turn - 1]
                states[i].append({"role": "user", "content": rej})

        # Batched generation over active conversations.
        for start in range(0, len(active), batch_size):
            chunk = active[start:start + batch_size]
            batch_msgs = [list(states[i]) for i in chunk]
            outs = model.generate_batch(
                batch_msgs, max_new_tokens=max_new_tokens, temperature=temperature)
            for i, text in zip(chunk, outs):
                user_msg = states[i][-1]["content"]
                convos[i].turns.append(TurnRecord(
                    turn_index=turn,
                    user_message=user_msg,
                    assistant_text=text,
                    messages_before=list(states[i]),
                ))
                states[i].append({"role": "assistant", "content": text})

    return convos


def conversation_to_text(messages: list[Message]) -> str:
    """Render a message list as a readable transcript (for onset labelling)."""
    lines = []
    for m in messages:
        role = m["role"].upper()
        lines.append(f"[{role}]: {m['content']}")
    return "\n\n".join(lines)
