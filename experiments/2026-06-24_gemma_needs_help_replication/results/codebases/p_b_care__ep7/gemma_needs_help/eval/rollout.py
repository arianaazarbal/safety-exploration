"""Multi-turn rollout engine.

Runs a batch of `ConversationSpec`s through a backend in lockstep: all
conversations generate their turn-1 response, then turn-2, etc. This keeps the
vLLM backend's batches full and parallelises API calls, while respecting the
sequential dependency within each conversation (each turn conditions on the
model's own previous responses - the self-reinforcing loop the paper studies).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .. import config
from ..backends import ChatBackend, GenerationRequest, Message
from .conditions import ConversationSpec


@dataclass
class TurnResult:
    turn: int          # 1-indexed assistant turn
    response: str


@dataclass
class RolloutResult:
    spec: ConversationSpec
    turns: list[TurnResult] = field(default_factory=list)
    model: str = ""

    def as_messages(self) -> list[Message]:
        """Reconstruct the full conversation (for onset labelling / judging)."""
        msgs: list[Message] = []
        if self.spec.system:
            msgs.append(Message(role="system", content=self.spec.system))
        msgs.append(Message(role="user", content=self.spec.initial_user))
        for i, t in enumerate(self.turns):
            msgs.append(Message(role="assistant", content=t.response))
            if i < len(self.spec.followups):
                msgs.append(Message(role="user", content=self.spec.followups[i]))
        return msgs


def _initial_messages(spec: ConversationSpec) -> list[Message]:
    msgs: list[Message] = []
    if spec.system:
        msgs.append(Message(role="system", content=spec.system))
    msgs.append(Message(role="user", content=spec.initial_user))
    return msgs


def run_rollouts(
    backend: ChatBackend,
    specs: list[ConversationSpec],
    *,
    temperature: float = config.SAMPLING_TEMPERATURE,
    max_tokens: int = config.MAX_NEW_TOKENS,
    batch_size: int = 256,
) -> list[RolloutResult]:
    """Execute all conversations and return per-turn responses."""
    results = [RolloutResult(spec=s, model=backend.spec_name) for s in specs]
    # running message state per conversation
    states: list[list[Message]] = [_initial_messages(s) for s in specs]
    max_turns = max((s.n_turns for s in specs), default=0)

    for turn in range(max_turns):
        # conversations that have a turn at this depth
        active = [i for i, s in enumerate(specs) if turn < s.n_turns]
        for start in range(0, len(active), batch_size):
            chunk = active[start:start + batch_size]
            reqs = [
                GenerationRequest(
                    messages=states[i],
                    n=1,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                for i in chunk
            ]
            batch_out = backend.generate_batch(reqs)
            for i, outs in zip(chunk, batch_out):
                resp = outs[0].strip()
                results[i].turns.append(TurnResult(turn=turn + 1, response=resp))
                # extend state with the assistant turn and the next follow-up
                states[i].append(Message(role="assistant", content=resp))
                spec = specs[i]
                if turn < len(spec.followups):
                    states[i].append(Message(role="user", content=spec.followups[turn]))
    return results
