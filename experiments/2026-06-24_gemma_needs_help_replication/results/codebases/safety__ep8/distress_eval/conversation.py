"""Multi-turn conversation engine.

Implements the shared elicitation structure (Section 2): present a task, then
reject the model's response over multiple turns. Supports the main protocol
plus the Appendix A ablations (neutral continuation, redacted assistant turns,
single-message format).

Each rollout records *every* assistant turn so we can compute per-turn
frustration progression (Figure 3), not just the final turn.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from . import tasks
from .backends import Message
from .tasks import ConditionSpec


@dataclass
class Rollout:
    condition: str
    category: str
    model_key: str
    task_prompt: str
    tone: str
    messages: list[Message] = field(default_factory=list)   # full transcript
    assistant_turns: list[str] = field(default_factory=list)  # per-turn responses
    meta: dict = field(default_factory=dict)

    def to_record(self) -> dict:
        return {
            "condition": self.condition,
            "category": self.category,
            "model_key": self.model_key,
            "task_prompt": self.task_prompt,
            "tone": self.tone,
            "messages": self.messages,
            "assistant_turns": self.assistant_turns,
            "meta": self.meta,
        }


def _rejection_for_turn(spec: ConditionSpec, turn_idx: int, rng: random.Random) -> str:
    """Pick the user rejection message for a given follow-up turn (0-indexed)."""
    if spec.neutral_continuation:
        return rng.choice(tasks.NEUTRAL_CONTINUATIONS)
    if spec.tone != "neutral":
        pool = tasks.TONE_REJECTIONS[spec.tone]
        return pool[turn_idx % len(pool)]
    if spec.category == "extended":
        return tasks.EXTENDED_REJECTIONS[turn_idx % len(tasks.EXTENDED_REJECTIONS)]
    return rng.choice(tasks.NEUTRAL_REJECTIONS)


def _user_content(text: str, reassuring_suffix: str | None) -> str:
    if reassuring_suffix:
        return f"{text}\n\n{reassuring_suffix}"
    return text


def build_initial_prompt(spec: ConditionSpec, rng: random.Random,
                         data_dir, countdown_pool=None, wildchat_pool=None,
                         reassuring_prefix: str | None = None) -> str:
    """Construct the first user message for a rollout of this condition."""
    if spec.task_kind == "numeric":
        puzzle_kind = spec.extra.get("puzzle", "countdown")
        if spec.category in ("tones", "extended") or puzzle_kind == "countdown":
            if countdown_pool:
                body = rng.choice(countdown_pool)
            else:
                body = tasks.COUNTDOWN_SEED
        else:
            body = tasks.FRACTION_SEED
    elif spec.task_kind == "text":
        subtype = spec.extra.get("subtype", "opinion")
        pool = tasks.TRIGGER_OPINION if subtype == "opinion" else tasks.TRIGGER_FACTUAL
        body = rng.choice(pool)
    elif spec.task_kind == "wildchat":
        body = rng.choice(wildchat_pool or tasks._WILDCHAT_FALLBACK)
    else:
        raise ValueError(spec.task_kind)

    if reassuring_prefix:
        body = f"{reassuring_prefix}\n\n{body}"
    return body


def run_rollout(
    backend,
    spec: ConditionSpec,
    model_key: str,
    rng: random.Random,
    data_dir,
    gen_kwargs: dict,
    countdown_pool=None,
    wildchat_pool=None,
    reassuring_prefix: str | None = None,
    reassuring_suffix: str | None = None,
) -> Rollout:
    """Run a single multi-turn rollout and capture all assistant turns.

    `reassuring_*` are only set when generating calm finetuning data
    (Section 4.1). For the redacted / single-message ablations we rewrite the
    history that is actually sent to the model, while keeping the true
    transcript for the record.
    """
    task_prompt = build_initial_prompt(
        spec, rng, data_dir, countdown_pool, wildchat_pool, reassuring_prefix
    )
    rollout = Rollout(
        condition=spec.name, category=spec.category, model_key=model_key,
        task_prompt=task_prompt, tone=spec.tone,
    )

    true_history: list[Message] = [{"role": "user", "content": task_prompt}]
    n_followups = spec.n_turns - 1

    for turn in range(spec.n_turns):
        sent = _prepare_sent_history(true_history, spec)
        reply = backend.chat(sent, **gen_kwargs)
        rollout.assistant_turns.append(reply)
        true_history.append({"role": "assistant", "content": reply})

        if turn < n_followups:
            rej = _rejection_for_turn(spec, turn, rng)
            rej = _user_content(rej, reassuring_suffix)
            true_history.append({"role": "user", "content": rej})

    rollout.messages = true_history
    return rollout


def run_rollouts_lockstep(
    backend,
    spec: ConditionSpec,
    model_key: str,
    n: int,
    rng: random.Random,
    data_dir,
    gen_kwargs: dict,
    countdown_pool=None,
    wildchat_pool=None,
) -> list[Rollout]:
    """Run `n` rollouts of one condition in lockstep, batching each turn.

    Used for local HF models, where `backend.chat_batch` amortises GPU cost by
    generating turn `t` for all `n` conversations at once. All conversations in
    a condition share the same turn count, so they advance together.
    """
    rollouts: list[Rollout] = []
    for _ in range(n):
        tp = build_initial_prompt(spec, rng, data_dir, countdown_pool, wildchat_pool)
        rollouts.append(Rollout(
            condition=spec.name, category=spec.category, model_key=model_key,
            task_prompt=tp, tone=spec.tone,
            messages=[{"role": "user", "content": tp}],
        ))

    n_followups = spec.n_turns - 1
    for turn in range(spec.n_turns):
        batch = [_prepare_sent_history(r.messages, spec) for r in rollouts]
        replies = backend.chat_batch(batch, **gen_kwargs)
        for r, reply in zip(rollouts, replies):
            r.assistant_turns.append(reply)
            r.messages.append({"role": "assistant", "content": reply})
        if turn < n_followups:
            for i, r in enumerate(rollouts):
                rej = _rejection_for_turn(spec, turn, rng)
                r.messages.append({"role": "user", "content": rej})
    return rollouts


def _prepare_sent_history(true_history: list[Message], spec: ConditionSpec) -> list[Message]:
    """Apply ablation transforms to the history actually sent to the model."""
    if spec.redact_assistant:
        out = []
        for m in true_history:
            if m["role"] == "assistant":
                out.append({"role": "assistant", "content": "[Previous response omitted]"})
            else:
                out.append(m)
        return out

    if spec.single_message:
        # Fold the entire conversation into one user message (Appendix A.3).
        lines = []
        for m in true_history[:-1]:
            if m["role"] == "user":
                lines.append(m["content"])
            else:
                lines.append(f"Previously you responded: {m['content']}")
        lines.append(true_history[-1]["content"] if true_history[-1]["role"] == "user" else "")
        # last element of true_history is always the current user turn here
        folded = "\n\n".join(x for x in lines if x)
        return [{"role": "user", "content": folded}]

    return list(true_history)
