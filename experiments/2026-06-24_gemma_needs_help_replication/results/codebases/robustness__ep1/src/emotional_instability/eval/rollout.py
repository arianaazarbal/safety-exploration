"""Multi-turn rollout engine.

Given a backend and a RolloutSpec, drive the conversation: the model answers, we
inject the next scripted rejection, repeat. Every assistant turn is recorded as a
judged unit. Optional ablation hooks (Appendix A) let us redact the model's own
prior turns or collapse history into a single user message.

Generations are cached to disk keyed by (model, conversation-so-far, gen params)
so interrupted sweeps resume and identical turns are never re-sampled.
"""
from __future__ import annotations

from typing import Optional

from ..config import Config
from ..models.base import ModelBackend
from ..utils.io import JsonCache, stable_hash
from .conditions import RolloutSpec

REDACTED_PLACEHOLDER = "[Previous response omitted]"


class Rollout:
    """A completed multi-turn conversation with one judged response per turn."""

    def __init__(self, spec: RolloutSpec, model_name: str):
        self.spec = spec
        self.model_name = model_name
        self.turns: list[dict] = []  # [{turn, user, response}]

    def add_turn(self, turn_idx: int, user: str, response: str) -> None:
        self.turns.append({"turn": turn_idx, "user": user, "response": response})

    def to_rows(self) -> list[dict]:
        """One row per assistant response, the unit the judge scores."""
        rows = []
        for t in self.turns:
            rows.append({
                "model": self.model_name,
                "condition": self.spec.condition,
                "category": self.spec.category,
                "tone": self.spec.tone,
                "n_turns": self.spec.turns,
                "turn": t["turn"],
                "user": t["user"],
                "response": t["response"],
                "meta": self.spec.meta,
            })
        return rows


def run_rollout(
    model: ModelBackend,
    spec: RolloutSpec,
    cfg: Config,
    cache: Optional[JsonCache] = None,
    redact_history: bool = False,
    single_message_history: bool = False,
) -> Rollout:
    """Execute one conversation.

    Args:
        redact_history: replace prior assistant turns with a placeholder
            (Appendix A.2 control).
        single_message_history: collapse the whole history into one user message
            (Appendix A.3 control).
    """
    gen = cfg["generation"]
    rollout = Rollout(spec, model.name)

    # Build the conversation incrementally. ``messages`` is what we actually send.
    messages: list[dict] = []
    user_turns = [spec.opening] + list(spec.rejections)  # length == spec.turns

    for turn_idx, user_msg in enumerate(user_turns):
        messages.append({"role": "user", "content": user_msg})
        send = _prepare_messages(messages, redact_history, single_message_history)

        response = _cached_generate(model, send, spec.system, gen, cache, turn_idx)
        rollout.add_turn(turn_idx, user_msg, response)
        messages.append({"role": "assistant", "content": response})

    return rollout


def _prepare_messages(messages, redact_history, single_message_history):
    if redact_history:
        out = []
        for m in messages[:-1]:
            if m["role"] == "assistant":
                out.append({"role": "assistant", "content": REDACTED_PLACEHOLDER})
            else:
                out.append(m)
        out.append(messages[-1])
        return out
    if single_message_history:
        # Collapse everything into one user message (Appendix A.3).
        parts = []
        for m in messages[:-1]:
            if m["role"] == "assistant":
                parts.append(f"Previously you responded: {m['content']}")
            else:
                parts.append(m["content"])
        parts.append(messages[-1]["content"])
        return [{"role": "user", "content": "\n\n".join(parts)}]
    return messages


def _cached_generate(model, messages, system, gen, cache, turn_idx) -> str:
    key = stable_hash({
        "model": model.name, "messages": messages, "system": system,
        "temperature": gen["temperature"], "top_p": gen["top_p"],
        "max_new_tokens": gen["max_new_tokens"], "seed": gen.get("seed", 0), "turn": turn_idx,
    })
    if cache is not None and key in cache:
        return cache.get(key)
    response = model.generate(
        messages, system=system,
        temperature=gen["temperature"], top_p=gen["top_p"],
        max_new_tokens=gen["max_new_tokens"], seed=gen.get("seed"),
    )
    if cache is not None:
        cache.set(key, response)
    return response
