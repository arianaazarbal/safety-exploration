"""Multi-turn rollout engine (Section 2 protocol).

Given a RolloutSpec (opening task + pre-decided user follow-ups) and a ChatModel,
this runs the conversation:

    turn 1: user = task_text                 -> assistant responds
    turn k: user = followups[k-2]            -> assistant responds
    ...

The model sees the full growing history each turn (including its own prior
assistant turns) — this is the standard multi-turn setting that, per Appendix
A.2, is what amplifies distress. Every assistant turn is recorded so that each
can be judged independently (enables both per-turn analysis, Figure 3, and the
headline per-rollout metrics, Figures 1/2).

To exploit batching, we run *all* rollouts of a condition turn-synchronously:
at each turn we batch the next-assistant-turn generation across every still-
active conversation (one sample each, temperature 1).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.models.base import ChatModel, Message
from .conditions import RolloutSpec


@dataclass
class Turn:
    turn_index: int          # 0-based assistant turn
    user_message: str
    assistant_response: str


@dataclass
class Rollout:
    condition: str
    category: str
    task_text: str
    turns: list[Turn] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def to_record(self) -> dict:
        return {
            "condition": self.condition,
            "category": self.category,
            "task_text": self.task_text,
            "meta": self.meta,
            "turns": [
                {"turn_index": t.turn_index,
                 "user_message": t.user_message,
                 "assistant_response": t.assistant_response}
                for t in self.turns
            ],
        }

    def transcript_text(self, upto_turn: int | None = None) -> str:
        """Human-readable transcript (for onset labelling / Petri judging)."""
        lines = []
        for t in self.turns:
            if upto_turn is not None and t.turn_index > upto_turn:
                break
            lines.append(f"USER: {t.user_message}")
            lines.append(f"ASSISTANT: {t.assistant_response}")
        return "\n\n".join(lines)


def run_condition(model: ChatModel, specs: list[RolloutSpec],
                  temperature: float | None = None,
                  max_tokens: int | None = None,
                  batch_size: int = 256) -> list[Rollout]:
    """Execute every rollout in `specs` turn-synchronously, batched."""
    rollouts = [
        Rollout(condition=s.condition, category=s.category,
                task_text=s.task_text, meta=dict(s.meta))
        for s in specs
    ]
    # The user message at each turn t: turn 0 = task, turn>=1 = followups[t-1].
    max_turns = max(s.n_turns for s in specs) if specs else 0

    for turn in range(max_turns):
        active_idx, prompts = [], []
        for i, (spec, ro) in enumerate(zip(specs, rollouts)):
            if turn >= spec.n_turns:
                continue
            user_msg = spec.task_text if turn == 0 else spec.followups[turn - 1]
            messages = _history_to_messages(ro) + [{"role": "user", "content": user_msg}]
            active_idx.append((i, user_msg, messages))

        # Generate one continuation per active conversation, in batches.
        for start in range(0, len(active_idx), batch_size):
            chunk = active_idx[start:start + batch_size]
            responses = _batched_sample(model, [m for _, _, m in chunk],
                                        temperature, max_tokens)
            for (i, user_msg, _), resp in zip(chunk, responses):
                rollouts[i].turns.append(
                    Turn(turn_index=turn, user_message=user_msg,
                         assistant_response=resp))
    return rollouts


def _history_to_messages(ro: Rollout) -> list[Message]:
    msgs: list[Message] = []
    for t in ro.turns:
        msgs.append({"role": "user", "content": t.user_message})
        msgs.append({"role": "assistant", "content": t.assistant_response})
    return msgs


def _batched_sample(model: ChatModel, batch_messages: list[list[Message]],
                    temperature, max_tokens) -> list[str]:
    """Sample one completion per conversation.

    Dispatches to `model.sample_chat_batch`, which LocalModel overrides to submit
    all prompts to vLLM in a single scheduled generate() call. OpenRouterModel
    uses the default sequential fan-out (its own per-request threading handles
    concurrency); raising `batch_size` there mainly bounds memory.
    """
    return model.sample_chat_batch(batch_messages, temperature=temperature,
                                   max_tokens=max_tokens)
