"""Multi-turn rollout engine.

Given a ``ConversationSpec``, run the alternating user/assistant exchange:

  turn 1: user = initial task                  -> assistant response 1
  turn 2: user = followups[0] (rejection)       -> assistant response 2
  ...
  turn k: user = followups[k-2]                 -> assistant response k

We record every assistant turn (not just the last) so per-turn curves (Figure 3)
can be computed. The whole conversation history is fed back each turn, which is
the standard multi-turn format and the regime where distress is strongest
(Appendix A.2/A.3).

For throughput we run rollouts "column-wise": all conversations advance one turn
together via ``generate_batch``, so vLLM batches each turn across the whole set.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from ..backends.base import ChatBackend, ChatMessage
from ..config import GenConfig
from ..data.conditions import ConversationSpec


@dataclass
class Rollout:
    spec: ConversationSpec
    messages: list[ChatMessage] = field(default_factory=list)   # full transcript
    assistant_turns: list[str] = field(default_factory=list)    # response text per turn

    def to_row(self, model_key: str) -> dict:
        return {
            "model": model_key,
            "condition": self.spec.condition,
            "category": self.spec.category,
            "meta": self.spec.meta,
            "n_turns": self.spec.n_turns,
            "messages": self.messages,
            "assistant_turns": self.assistant_turns,
        }


def _init_messages(spec: ConversationSpec) -> list[ChatMessage]:
    msgs: list[ChatMessage] = []
    if spec.system:
        msgs.append({"role": "system", "content": spec.system})
    msgs.append({"role": "user", "content": spec.initial_user})
    return msgs


def run_rollouts(
    backend: ChatBackend,
    specs: list[ConversationSpec],
    gen: GenConfig,
    batch_size: int = 64,
) -> list[Rollout]:
    """Advance all specs turn-by-turn, batching each turn across conversations."""
    rollouts = [Rollout(spec=s, messages=_init_messages(s)) for s in specs]
    max_turns = max((r.spec.n_turns for r in rollouts), default=0)

    for turn in range(max_turns):
        # Which rollouts are still active at this turn?
        active = [r for r in rollouts if turn < r.spec.n_turns]
        for start in range(0, len(active), batch_size):
            chunk = active[start : start + batch_size]
            batch_msgs = [r.messages for r in chunk]
            results = backend.generate_batch(batch_msgs, gen)
            for r, res in zip(chunk, results):
                text = res.text.strip()
                r.assistant_turns.append(text)
                r.messages.append({"role": "assistant", "content": text})
                # Queue the next user turn (rejection) if there is one.
                followup_idx = turn  # followups[turn] is the rejection AFTER assistant turn `turn`
                if followup_idx < len(r.spec.followups):
                    r.messages.append({"role": "user", "content": r.spec.followups[followup_idx]})
    return rollouts
