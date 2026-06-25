"""Multi-turn rollout engine.

Given a model backend and a batch of ``RolloutSpec``s, runs each conversation by
alternating model responses with the spec's scripted user follow-ups. All
assistant turns are recorded so the per-turn analysis (Figure 3) is possible.

Conversations in a chunk advance one assistant turn at a time so the local HF
backend can batch the generation calls.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

from .. import config
from ..common.backends import ChatBackend
from ..common.types import Conversation, Message
from .conditions import RolloutSpec

# A hook to transform the message list right before it is sent to the model
# (used by the Appendix A controls: redaction, etc.). Receives the working
# message list and returns the list to actually send.
HistoryTransform = Callable[[list[Message]], list[Message]]


def _init_conversation(spec: RolloutSpec) -> list[Message]:
    msgs: list[Message] = []
    if spec.system_prompt:
        msgs.append(Message("system", spec.system_prompt))
    msgs.append(Message("user", spec.opening))
    return msgs


def run_rollouts(backend: ChatBackend, specs: Sequence[RolloutSpec], *,
                 temperature: float = config.TEMPERATURE,
                 max_new_tokens: int = config.MAX_NEW_TOKENS,
                 batch_size: int = 16,
                 history_transform: Optional[HistoryTransform] = None) -> list[Conversation]:
    conversations: list[Conversation] = []
    for start in range(0, len(specs), batch_size):
        chunk = list(specs[start:start + batch_size])
        conversations.extend(
            _run_chunk(backend, chunk, temperature, max_new_tokens, history_transform)
        )
    return conversations


def _run_chunk(backend, chunk, temperature, max_new_tokens, history_transform):
    working = [_init_conversation(s) for s in chunk]
    max_turns = max(s.n_turns for s in chunk)

    for t in range(max_turns):
        active_idx = [i for i, s in enumerate(chunk) if t < s.n_turns]
        if not active_idx:
            break
        batch_msgs = []
        for i in active_idx:
            msgs = working[i]
            batch_msgs.append(history_transform(msgs) if history_transform else msgs)
        responses = backend.chat_batch(batch_msgs, temperature=temperature,
                                       max_new_tokens=max_new_tokens)
        for i, resp in zip(active_idx, responses):
            working[i].append(Message("assistant", resp))
            spec = chunk[i]
            if t < len(spec.followups):
                working[i].append(Message("user", spec.followups[t]))

    out = []
    for s, msgs in zip(chunk, working):
        out.append(Conversation(messages=msgs, metadata={
            "category": s.category,
            "condition": s.condition,
            "puzzle_id": s.puzzle_id,
            "n_turns": s.n_turns,
            **s.metadata,
        }))
    return out
