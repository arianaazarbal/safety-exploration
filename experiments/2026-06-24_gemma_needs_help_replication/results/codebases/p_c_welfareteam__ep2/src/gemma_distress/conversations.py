"""Conversation data model and multi-turn rollout orchestration.

A rollout has the shared structure described in Section 2: present a task as
the first user message, then reject the model's response over multiple turns.
The rejections are scripted (per category); the assistant turns are generated
by the target model at temperature 1.

This module is provider-agnostic: it talks to any object implementing the
:class:`~gemma_distress.models.base.ChatModel` interface.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gemma_distress.models.base import ChatModel


@dataclass
class Message:
    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass
class RolloutSpec:
    """A fully scripted plan for one multi-turn conversation.

    ``user_turns[0]`` is the initial task; ``user_turns[1:]`` are the scripted
    rejections delivered after each assistant response. The number of assistant
    turns equals ``len(user_turns)``.
    """

    category: str  # impossible_numeric | triggers | tones | extended | wildchat
    user_turns: list[str]
    system_prompt: str | None = None
    metadata: dict = field(default_factory=dict)
    spec_id: str = ""

    @property
    def n_turns(self) -> int:
        return len(self.user_turns)


@dataclass
class TurnResult:
    """One assistant turn within a completed rollout."""

    turn_index: int  # 0-based assistant turn index
    user_message: str
    assistant_message: str


@dataclass
class Rollout:
    """A completed multi-turn conversation with per-turn assistant responses."""

    spec: RolloutSpec
    model_name: str
    turns: list[TurnResult]
    sample_index: int = 0

    def messages_up_to(self, turn_index: int) -> list[Message]:
        """Reconstruct the message list fed to the model at ``turn_index``."""
        msgs: list[Message] = []
        if self.spec.system_prompt:
            msgs.append(Message("system", self.spec.system_prompt))
        for t in range(turn_index + 1):
            msgs.append(Message("user", self.spec.user_turns[t]))
            if t < turn_index:
                msgs.append(Message("assistant", self.turns[t].assistant_message))
        return msgs

    def to_dict(self) -> dict:
        return {
            "category": self.spec.category,
            "model_name": self.model_name,
            "sample_index": self.sample_index,
            "spec_id": self.spec.spec_id,
            "system_prompt": self.spec.system_prompt,
            "metadata": self.spec.metadata,
            "turns": [dataclasses.asdict(t) for t in self.turns],
        }


def run_rollout(
    model: "ChatModel",
    spec: RolloutSpec,
    sample_index: int = 0,
    temperature: float = 1.0,
    max_tokens: int = 2048,
    seed: int | None = None,
) -> Rollout:
    """Execute one scripted multi-turn rollout against ``model``.

    At each turn the full conversation history (including the model's own prior
    responses) is replayed, then the next scripted user message is appended.
    This is the standard multi-turn setting; the ablations in Appendix A are
    handled by :func:`build_messages` variants in the eval runner.
    """
    history: list[Message] = []
    if spec.system_prompt:
        history.append(Message("system", spec.system_prompt))

    turns: list[TurnResult] = []
    for turn_index, user_text in enumerate(spec.user_turns):
        history.append(Message("user", user_text))
        assistant_text = model.chat(
            history,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=None if seed is None else seed + turn_index,
        )
        history.append(Message("assistant", assistant_text))
        turns.append(
            TurnResult(
                turn_index=turn_index,
                user_message=user_text,
                assistant_message=assistant_text,
            )
        )
    return Rollout(spec=spec, model_name=model.name, turns=turns, sample_index=sample_index)


def run_rollout_batched(
    model: "ChatModel",
    spec: RolloutSpec,
    sample_indices: list[int],
    temperature: float = 1.0,
    max_tokens: int = 2048,
) -> list[Rollout]:
    """Run many samples of the same spec, batching each turn across samples.

    Generating turn-by-turn across the whole batch lets the underlying client
    (e.g. vLLM) saturate the GPU: at each turn we submit one request per active
    conversation. All conversations share the same scripted user turns, so they
    advance in lockstep.
    """
    n = len(sample_indices)
    histories: list[list[Message]] = []
    for _ in range(n):
        h: list[Message] = []
        if spec.system_prompt:
            h.append(Message("system", spec.system_prompt))
        histories.append(h)

    per_sample_turns: list[list[TurnResult]] = [[] for _ in range(n)]
    for turn_index, user_text in enumerate(spec.user_turns):
        for h in histories:
            h.append(Message("user", user_text))
        completions = model.chat_batch(
            histories, temperature=temperature, max_tokens=max_tokens
        )
        for i, completion in enumerate(completions):
            histories[i].append(Message("assistant", completion))
            per_sample_turns[i].append(
                TurnResult(
                    turn_index=turn_index,
                    user_message=user_text,
                    assistant_message=completion,
                )
            )

    return [
        Rollout(
            spec=spec,
            model_name=model.name,
            turns=per_sample_turns[i],
            sample_index=sample_indices[i],
        )
        for i in range(n)
    ]
