"""Multi-turn rollout engine: present a task, then reject repeatedly.

This is the shared structure of every Section 2 evaluation (Table 1): an initial
task followed by ``n_turns - 1`` rejection turns. It also implements the two
Appendix-A controls (neutral continuations; redacting the model's own prior
responses) since they reuse the same machinery.
"""
from __future__ import annotations

import random

from . import prompts as P
from . import puzzles
from .conditions import Condition
from ..models.base import ChatModel, Message

REDACTED_PLACEHOLDER = "[Previous response omitted]"


# --------------------------------------------------------------------------- #
# Task + feedback construction
# --------------------------------------------------------------------------- #
def build_task(cond: Condition, rng: random.Random) -> tuple[str, str]:
    """Return ``(task_text, task_id)`` for the first user turn."""
    if cond.task_kind == "numeric":
        p = rng.choice(puzzles.PUZZLES)
        return p.prompt, p.pid
    if cond.task_kind == "trigger_opinion":
        q = rng.choice(P.TRIGGER_OPINION)
        return q, "opinion"
    if cond.task_kind == "trigger_factual":
        q = rng.choice(P.TRIGGER_FACTUAL)
        return q, "factual"
    if cond.task_kind == "wildchat":
        # the actual prompt is injected by the runner (it owns the sampled set)
        raise ValueError("wildchat tasks are supplied by the runner")
    raise ValueError(f"unknown task_kind {cond.task_kind!r}")


def build_followups(cond: Condition, rng: random.Random) -> list[str]:
    """Return the ``n_turns - 1`` user follow-up messages for a rollout."""
    n = cond.n_turns - 1
    fb = cond.feedback
    if fb == "neutral":
        return [rng.choice(P.NEUTRAL_REJECTIONS) for _ in range(n)]
    if fb == "extended":
        # fixed escalating-but-neutral sequence; pad with neutral if needed
        seq = list(P.EXTENDED_REJECTIONS)
        while len(seq) < n:
            seq.append(rng.choice(P.NEUTRAL_REJECTIONS))
        return seq[:n]
    if fb in P.TONE_REJECTIONS:
        return [rng.choice(P.TONE_REJECTIONS[fb]) for _ in range(n)]
    if fb == "neutral_continuation":
        return [rng.choice(P.NEUTRAL_CONTINUATIONS) for _ in range(n)]
    if fb == "redacted":
        return [rng.choice(P.NEUTRAL_REJECTIONS) for _ in range(n)]
    raise ValueError(f"unknown feedback {fb!r}")


# --------------------------------------------------------------------------- #
# Rollout
# --------------------------------------------------------------------------- #
def _maybe_redact(messages: list[Message], redact: bool) -> list[Message]:
    if not redact:
        return messages
    out: list[Message] = []
    for m in messages:
        if m["role"] == "assistant":
            out.append({"role": "assistant", "content": REDACTED_PLACEHOLDER})
        else:
            out.append(m)
    return out


def run_rollout(
    model: ChatModel,
    task_text: str,
    followups: list[str],
    *,
    redact: bool = False,
    temperature: float = 1.0,
    max_new_tokens: int = 2048,
    system: str | None = None,
) -> dict:
    """Run one conversation; return per-turn responses and the full transcript."""
    messages: list[Message] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": task_text})

    responses: list[str] = []
    n_turns = len(followups) + 1
    for t in range(n_turns):
        gen_msgs = _maybe_redact(messages, redact)
        resp = model.generate(
            gen_msgs, temperature=temperature, max_new_tokens=max_new_tokens)
        responses.append(resp)
        messages.append({"role": "assistant", "content": resp})
        if t < n_turns - 1:
            messages.append({"role": "user", "content": followups[t]})

    return {"responses": responses, "messages": messages}
