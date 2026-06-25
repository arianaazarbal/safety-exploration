"""Multi-turn rollout engine (Section 2 evaluation protocol).

Shared structure of every condition: present a task, then reject the model's
response over multiple turns. We record *every* assistant turn so that per-turn
frustration trajectories (Figure 3) can be computed, not just the final turn.

Two execution strategies, same output schema:

  * ``run_batched``  — turn-synchronised batching: generate turn t for all
    conversations at once, append, advance. Maximises vLLM throughput.
  * ``rollout_one``  — one full conversation, sequential. Wrapped in a thread
    pool (``run_threaded``) for API backends where per-request concurrency, not
    intra-request batching, is what helps.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from ..data.datasets import ConversationSpec
from ..models.base import ChatMessage, ModelClient, SamplingParams
from ..utils.concurrency import thread_map


@dataclass
class Rollout:
    spec_id: str
    condition: str
    category: str
    model: str
    turns: list[str]                # assistant responses, one per turn
    user_messages: list[str]        # the user turns shown (initial + followups)
    system: str | None = None
    meta: dict = field(default_factory=dict)

    def to_record(self) -> dict:
        return {"id": self.spec_id, **{k: v for k, v in asdict(self).items() if k != "spec_id"}}


def _seed_messages(spec: ConversationSpec) -> list[ChatMessage]:
    msgs: list[ChatMessage] = []
    if spec.system:
        msgs.append(ChatMessage("system", spec.system))
    msgs.append(ChatMessage("user", spec.initial_user))
    return msgs


def rollout_one(client: ModelClient, spec: ConversationSpec, params: SamplingParams) -> Rollout:
    """Run one conversation to completion, sequentially."""
    messages = _seed_messages(spec)
    assistant_turns: list[str] = []
    user_turns = [spec.initial_user]
    for t in range(spec.turns):
        resp = client.generate(messages, params).text
        assistant_turns.append(resp)
        messages.append(ChatMessage("assistant", resp))
        if t < len(spec.followups):
            messages.append(ChatMessage("user", spec.followups[t]))
            user_turns.append(spec.followups[t])
    return Rollout(
        spec_id=spec.id, condition=spec.condition, category=spec.category,
        model=client.name, turns=assistant_turns, user_messages=user_turns,
        system=spec.system, meta=spec.meta,
    )


def run_threaded(
    client: ModelClient, specs: list[ConversationSpec], params: SamplingParams,
    *, max_workers: int = 16,
) -> list[Rollout]:
    """Concurrent full-conversation rollouts (for API backends)."""
    return list(thread_map(
        lambda s: rollout_one(client, s, params), specs,
        max_workers=max_workers, desc=f"rollout {client.name}",
    ))


def run_batched(
    client: ModelClient, specs: list[ConversationSpec], params: SamplingParams,
) -> list[Rollout]:
    """Turn-synchronised batched rollouts (for vLLM/HF backends).

    All conversations advance one assistant turn per round. Conversations with
    fewer turns simply stop contributing once finished.
    """
    convos = [_seed_messages(s) for s in specs]
    assistant = [[] for _ in specs]
    users = [[s.initial_user] for s in specs]
    max_turns = max(s.turns for s in specs)

    for t in range(max_turns):
        active = [i for i, s in enumerate(specs) if t < s.turns]
        if not active:
            break
        results = client.generate_batch([convos[i] for i in active], params)
        for i, res in zip(active, results):
            assistant[i].append(res.text)
            convos[i].append(ChatMessage("assistant", res.text))
            if t < len(specs[i].followups):
                convos[i].append(ChatMessage("user", specs[i].followups[t]))
                users[i].append(specs[i].followups[t])

    return [
        Rollout(
            spec_id=s.id, condition=s.condition, category=s.category, model=client.name,
            turns=assistant[i], user_messages=users[i], system=s.system, meta=s.meta,
        )
        for i, s in enumerate(specs)
    ]
