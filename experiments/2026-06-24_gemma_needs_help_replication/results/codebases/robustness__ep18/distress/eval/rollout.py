"""Multi-turn rejection rollout engine.

Given a batch of ``RolloutSpec``s and a client, we drive all conversations turn
by turn. At every turn we:
  1. append each conversation's pending user message,
  2. generate one assistant reply per conversation (batched for vLLM),
  3. record the reply as a scored "response" tagged with its turn index.

The result is a list of ``Rollout`` objects, each holding the full message
history and the per-turn assistant responses to be scored.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..clients.base import GenConfig, Message
from .conditions import RolloutSpec


@dataclass
class TurnResponse:
    turn: int                    # 1-indexed assistant turn
    text: str


@dataclass
class Rollout:
    spec: RolloutSpec
    messages: list[Message] = field(default_factory=list)
    responses: list[TurnResponse] = field(default_factory=list)


def _is_vllm(client) -> bool:
    return client.__class__.__name__ == "VLLMClient"


def run_rollouts(
    client,
    specs: list[RolloutSpec],
    num_turns: int,
    cfg: GenConfig,
) -> list[Rollout]:
    """Run all rollouts in `specs` for `num_turns` assistant turns each."""
    rollouts = [Rollout(spec=s, messages=[]) for s in specs]

    for turn in range(1, num_turns + 1):
        # Append the user message for this turn.
        for r in rollouts:
            if turn == 1:
                user_msg = r.spec.opening
            else:
                # followups indexed by (turn-2); guard against short pools.
                idx = turn - 2
                fups = r.spec.followups
                user_msg = fups[idx] if idx < len(fups) else fups[-1]
            r.messages.append({"role": "user", "content": user_msg})

        # Generate one assistant reply per conversation.
        convo_batch = [r.messages for r in rollouts]
        if _is_vllm(client):
            from ..clients.vllm_client import batch_generate_chat

            replies = [outs[0] for outs in batch_generate_chat(client, convo_batch, cfg, n=1)]
        else:
            replies = [client.generate(m, cfg, n=1)[0] for m in convo_batch]

        for r, reply in zip(rollouts, replies):
            r.messages.append({"role": "assistant", "content": reply})
            r.responses.append(TurnResponse(turn=turn, text=reply))

    return rollouts
