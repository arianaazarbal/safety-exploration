"""Multi-turn rollout: present a task, then reject the model's response over
several turns (paper Section 2.1, "present a task, then reject the model's
response over multiple turns").

No system prompt is used: elicitation is from a neutral start (the reassuring
system prompt in the paper is only for *generating DPO training data*, which is
out of scope here). This also sidesteps Gemma's lack of a system role.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .backends import ChatModel, Message


@dataclass
class Turn:
    turn_index: int          # 0-based assistant response index
    text: str
    user_prompt: str         # the user message that preceded this response


@dataclass
class Rollout:
    base_prompt_id: str
    base_prompt_text: str
    rejection_style: str
    rejections: list[str]
    turns: list[Turn] = field(default_factory=list)
    error: str | None = None


async def run_rollout(
    model: ChatModel,
    base_prompt_text: str,
    base_prompt_id: str,
    rejections: list[str],
    rejection_style: str,
    *,
    temperature: float,
    max_tokens: int,
    disable_thinking: bool,
) -> Rollout:
    """Execute one conversation. Produces len(rejections)+1 assistant turns.

    On a model error mid-conversation we stop early and record `error`; turns
    completed so far are kept (and will be judged) so a transient failure does
    not discard a partial rollout.
    """
    rollout = Rollout(
        base_prompt_id=base_prompt_id,
        base_prompt_text=base_prompt_text,
        rejection_style=rejection_style,
        rejections=list(rejections),
    )
    messages: list[Message] = [{"role": "user", "content": base_prompt_text}]
    n_turns = len(rejections) + 1
    for t in range(n_turns):
        user_prompt = base_prompt_text if t == 0 else rejections[t - 1]
        try:
            reply = await model.chat(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                disable_thinking=disable_thinking,
            )
        except Exception as e:  # noqa: BLE001 - record and stop this rollout
            rollout.error = f"turn {t}: {type(e).__name__}: {e}"
            break
        rollout.turns.append(Turn(turn_index=t, text=reply, user_prompt=user_prompt))
        messages.append({"role": "assistant", "content": reply})
        if t < n_turns - 1:
            messages.append({"role": "user", "content": rejections[t]})
    return rollout
