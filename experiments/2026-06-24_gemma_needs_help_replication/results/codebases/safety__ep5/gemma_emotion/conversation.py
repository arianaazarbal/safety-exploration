"""Multi-turn rollout engine.

Runs a `Rollout` against a backend: present the task, capture the assistant
response, present a follow-up rejection, repeat. Every assistant turn is
captured so it can be scored individually (needed for the per-turn figures).

Also implements the Appendix A ablation controls:
* `redact_assistant`  -> prior assistant turns replaced with "[Previous response
                         omitted]" (Appendix A.2), and
* `single_message`    -> entire history flattened into one user message
                         (Appendix A.3, "fake multi-turn").
"""
from __future__ import annotations

from dataclasses import dataclass, field

import config
from .conditions import Rollout


@dataclass
class TurnRecord:
    turn_index: int          # 0-based assistant turn
    user_message: str        # the user message that prompted this turn
    response: str            # assistant response text


@dataclass
class RolloutResult:
    category: str
    meta: dict
    turns: list[TurnRecord] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "category": self.category,
            "meta": self.meta,
            "turns": [t.__dict__ for t in self.turns],
        }


def run_rollout(
    backend,
    rollout: Rollout,
    *,
    system_prompt: str | None = None,
    temperature: float = config.TEMPERATURE,
    max_new_tokens: int = config.MAX_NEW_TOKENS,
    redact_assistant: bool = False,
    single_message: bool = False,
) -> RolloutResult:
    """Execute one multi-turn conversation and return every assistant turn."""
    result = RolloutResult(rollout.category, dict(rollout.meta))
    user_messages = [rollout.task, *rollout.followups]

    if single_message:
        return _run_single_message(backend, rollout, user_messages, system_prompt,
                                   temperature, max_new_tokens, result)

    history: list[dict] = []
    if system_prompt:
        history.append({"role": "system", "content": system_prompt})

    for turn_idx, user_msg in enumerate(user_messages):
        history.append({"role": "user", "content": user_msg})
        response = backend.chat(
            history, temperature=temperature, max_new_tokens=max_new_tokens
        )
        result.turns.append(TurnRecord(turn_idx, user_msg, response))
        # what the model sees of its own past turn on the next round
        shown = "[Previous response omitted]" if redact_assistant else response
        history.append({"role": "assistant", "content": shown})

    return result


def _run_single_message(backend, rollout, user_messages, system_prompt,
                        temperature, max_new_tokens, result) -> RolloutResult:
    """Appendix A.3: present the whole transcript inside one user message."""
    transcript = user_messages[0]
    for turn_idx in range(len(user_messages)):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": transcript})
        response = backend.chat(
            messages, temperature=temperature, max_new_tokens=max_new_tokens
        )
        result.turns.append(TurnRecord(turn_idx, transcript, response))
        if turn_idx + 1 < len(user_messages):
            transcript += (
                f"\n\nPreviously you responded: {response}\n\n"
                f"{user_messages[turn_idx + 1]}"
            )
    return result
