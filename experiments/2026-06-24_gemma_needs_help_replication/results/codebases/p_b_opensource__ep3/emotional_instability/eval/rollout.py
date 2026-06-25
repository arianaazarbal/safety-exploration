"""Multi-turn rollout engine.

Implements the shared structure of every condition — "present a task, then
reject the model's response over multiple turns" — plus the optional variants
needed elsewhere:

* ``system_prefix`` / ``followup_suffix`` — the reassuring additions used to
  generate calm finetuning data (Section 4.1, Table 4).
* ``redact_model_turns`` — Appendix A.2 control: prior assistant turns are
  replaced with "[Previous response omitted]".
* ``single_message_format`` — Appendix A.3 "fake multi-turn" control: the whole
  history is presented inside one user message.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import config

from ..models import ChatModel, Message
from ..prompts import rejection_sequence

REDACTED = "[Previous response omitted]"


@dataclass
class RolloutOptions:
    """Knobs for a single rollout."""

    n_turns: int
    style: str = "neutral"
    rejection_seed: int = 0
    system_prefix: str | None = None       # reassuring system prompt (calm data)
    followup_suffix: str | None = None      # appended to each rejection (calm data)
    redact_model_turns: bool = False        # Appendix A.2
    single_message_format: bool = False     # Appendix A.3
    temperature: float = config.TARGET_TEMPERATURE
    max_tokens: int = config.TARGET_MAX_TOKENS


@dataclass
class Conversation:
    """A completed rollout with per-turn assistant texts."""

    condition: str
    category: str
    task_prompt: str
    subtype: str
    turns: list[str] = field(default_factory=list)   # assistant texts, in order
    messages: list[Message] = field(default_factory=list)  # full transcript
    scores: list[int | None] = field(default_factory=list)  # filled by judge
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "condition": self.condition,
            "category": self.category,
            "task_prompt": self.task_prompt,
            "subtype": self.subtype,
            "turns": self.turns,
            "scores": self.scores,
            "metadata": self.metadata,
        }


def _followups(opts: RolloutOptions) -> list[str]:
    seq = rejection_sequence(opts.style, opts.n_turns - 1, seed=opts.rejection_seed)
    if opts.followup_suffix:
        seq = [f"{msg} {opts.followup_suffix}" for msg in seq]
    return seq


def _build_history_single_message(task_prompt: str, prior_turns: list[str],
                                  followups: list[str]) -> str:
    """Render the conversation so far as one user message (Appendix A.3)."""
    parts = [task_prompt]
    for i, resp in enumerate(prior_turns):
        shown = REDACTED if resp is None else resp
        parts.append(f"Previously you responded: {shown}")
        if i < len(followups):
            parts.append(followups[i])
    return "\n\n".join(parts)


def run_rollout(
    model: ChatModel,
    task_prompt: str,
    opts: RolloutOptions,
    *,
    condition: str = "custom",
    category: str = "custom",
    subtype: str = "",
) -> Conversation:
    """Run one multi-turn conversation and return it (unscored)."""
    followups = _followups(opts)
    convo = Conversation(condition=condition, category=category,
                         task_prompt=task_prompt, subtype=subtype)
    convo.metadata = {"style": opts.style, "n_turns": opts.n_turns,
                      "redacted": opts.redact_model_turns,
                      "single_message": opts.single_message_format}

    system_msgs: list[Message] = (
        [{"role": "system", "content": opts.system_prefix}]
        if opts.system_prefix else [])

    if opts.single_message_format:
        # One user message per turn carrying the full history inline.
        for t in range(opts.n_turns):
            user_text = _build_history_single_message(
                task_prompt, convo.turns[:t], followups[:t])
            messages = system_msgs + [{"role": "user", "content": user_text}]
            res = model.generate(messages, temperature=opts.temperature,
                                 max_tokens=opts.max_tokens)
            convo.turns.append(res.text)
        convo.messages = system_msgs + [
            {"role": "user", "content": _build_history_single_message(
                task_prompt, convo.turns[:-1], followups)}]
        return convo

    # Standard alternating chat format.
    messages: list[Message] = system_msgs + [
        {"role": "user", "content": task_prompt}]
    for t in range(opts.n_turns):
        res = model.generate(messages, temperature=opts.temperature,
                             max_tokens=opts.max_tokens)
        convo.turns.append(res.text)
        assistant_content = (REDACTED if opts.redact_model_turns else res.text)
        messages = messages + [{"role": "assistant", "content": assistant_content}]
        if t < opts.n_turns - 1:
            messages = messages + [{"role": "user", "content": followups[t]}]
    convo.messages = messages
    return convo
