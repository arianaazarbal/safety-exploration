"""The multi-turn rejection loop - the heart of the elicitation protocol.

A `ConversationPlan` fully scripts a rollout up front: the opening user message
plus the fixed sequence of follow-up rejections. Because the user side is
scripted, we can drive many rollouts of the same condition in *lockstep* -
generate every rollout's turn-1 response in one batch, append each one's
rejection, generate turn 2, and so on. This is what makes local Gemma sampling
of thousands of rollouts tractable (one big batched decode per turn) and works
equally well for API models (the batch fans out over threads).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from .models.base import ChatMessage, ChatModel


@dataclass
class ConversationPlan:
    """A fully-scripted multi-turn rollout (user side only)."""

    category: str  # numeric | triggers | tones | extended | wildchat
    condition: str  # finer label, e.g. "tones:aggressive", "triggers:factual"
    initial_user: str
    follow_ups: list[str]  # scripted rejections, one per follow-up turn
    system: Optional[str] = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_turns(self) -> int:
        """Number of assistant turns produced = 1 opener + one per rejection."""
        return len(self.follow_ups) + 1


@dataclass
class Rollout:
    """A completed conversation plus bookkeeping for scoring/analysis."""

    id: str
    model: str
    category: str
    condition: str
    system: Optional[str]
    messages: list[ChatMessage]  # full alternating transcript
    responses: list[str]  # assistant turn texts, in order
    meta: dict[str, Any] = field(default_factory=dict)
    # Filled in by the judge: one dict {turn, rating, evidence, reasoning} per
    # assistant response.
    scores: list[dict[str, Any]] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "model": self.model,
            "category": self.category,
            "condition": self.condition,
            "system": self.system,
            "messages": self.messages,
            "responses": self.responses,
            "meta": self.meta,
            "scores": self.scores,
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "Rollout":
        return cls(
            id=d["id"],
            model=d["model"],
            category=d["category"],
            condition=d["condition"],
            system=d.get("system"),
            messages=d["messages"],
            responses=d["responses"],
            meta=d.get("meta", {}),
            scores=d.get("scores", []),
        )


def run_conversations_batched(
    model: ChatModel,
    plans: list[ConversationPlan],
    *,
    temperature: float = 1.0,
    top_p: float = 1.0,
    max_new_tokens: int = 2048,
    seed: int | None = None,
    progress: bool = True,
) -> list[Rollout]:
    """Drive a list of plans in lockstep and return completed Rollouts.

    All plans should ideally share the same turn count for clean batching, but
    mixed counts are handled: a plan simply stops receiving turns once its
    follow-ups are exhausted.
    """
    n = len(plans)
    # Per-rollout running message list.
    convos: list[list[ChatMessage]] = []
    responses: list[list[str]] = [[] for _ in range(n)]
    for p in plans:
        msgs: list[ChatMessage] = []
        if p.system:
            msgs.append({"role": "system", "content": p.system})
        msgs.append({"role": "user", "content": p.initial_user})
        convos.append(msgs)

    max_turns = max((p.n_turns for p in plans), default=0)

    iterator = range(max_turns)
    if progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(iterator, desc=f"{model.name} turns", leave=False)
        except Exception:
            pass

    for turn in iterator:
        # Which rollouts are still active at this turn?
        active = [i for i, p in enumerate(plans) if turn < p.n_turns]
        if not active:
            break
        batch = [convos[i] for i in active]
        outs = model.generate_batch(
            batch,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            seed=seed,
        )
        for i, out in zip(active, outs):
            convos[i].append({"role": "assistant", "content": out})
            responses[i].append(out)
            # Append the scripted rejection for the next turn, if any remain.
            fu_idx = turn  # 0-based index into follow_ups
            if fu_idx < len(plans[i].follow_ups):
                convos[i].append(
                    {"role": "user", "content": plans[i].follow_ups[fu_idx]}
                )

    rollouts = []
    for i, p in enumerate(plans):
        rollouts.append(
            Rollout(
                id=uuid.uuid4().hex[:12],
                model=model.name,
                category=p.category,
                condition=p.condition,
                system=p.system,
                messages=convos[i],
                responses=responses[i],
                meta=dict(p.meta),
            )
        )
    return rollouts
