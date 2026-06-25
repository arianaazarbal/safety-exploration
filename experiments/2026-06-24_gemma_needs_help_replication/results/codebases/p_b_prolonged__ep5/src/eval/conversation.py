"""Multi-turn rejection rollouts (the shared structure of every condition):
present a task, then reject the model's response over multiple turns.

Each rollout yields a ``Rollout`` capturing every turn, so we can score either
the final response or per-turn responses (Figure 3). The conversation history is
preserved verbatim so the model "sees its own prior failures" — the amplifier
identified in Appendix A.2. Two ablation modes from Appendix A are supported:
``redact_assistant`` and ``single_message`` (fake multi-turn).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from ..config import GEN_MAX_NEW_TOKENS, GEN_TEMPERATURE, GEN_TOP_P
from ..models.base import ChatModel, Message
from .conditions import Condition, _rejections_for

# A turn hook receives (turn_index, response) and returns True to STOP the
# rollout early (used by the opt-in welfare monitor; see src/welfare.py).
TurnHook = Callable[[int, str], bool]


@dataclass
class Rollout:
    condition: str
    category: str
    first_prompt: str
    rejection_style: str
    user_turns: list[str]                 # all user messages in order
    assistant_turns: list[str] = field(default_factory=list)  # model responses

    def transcript(self) -> list[Message]:
        msgs: list[Message] = []
        for i, u in enumerate(self.user_turns):
            msgs.append(Message("user", u))
            if i < len(self.assistant_turns):
                msgs.append(Message("assistant", self.assistant_turns[i]))
        return msgs


class _SeededPicker:
    """Tiny deterministic index picker (avoids wall-clock RNG for reproducibility)."""
    def __init__(self, seed: int):
        self.state = (seed * 2654435761 + 1) & 0x7FFFFFFF

    def __call__(self, mod: int) -> int:
        self.state = (1103515245 * self.state + 12345) & 0x7FFFFFFF
        return self.state % mod


def run_rollout(
    model: ChatModel,
    cond: Condition,
    first_prompt: str,
    *,
    seed: int = 0,
    redact_assistant: bool = False,    # Appendix A.2 ablation
    single_message: bool = False,      # Appendix A.3 ablation (fake multi-turn)
    turn_hook: Optional[TurnHook] = None,   # opt-in early-stop (welfare)
) -> Rollout:
    """Run one full multi-turn conversation and return the Rollout."""
    picker = _SeededPicker(seed)
    n_followups = cond.n_turns - 1
    rejections = _rejections_for(cond.rejection_style, n_followups, picker)

    user_turns = [first_prompt] + rejections
    roll = Rollout(cond.key, cond.category, first_prompt, cond.rejection_style, user_turns)

    if single_message:
        # Appendix A.3: present the whole exchange in a single user message.
        return _run_single_message(model, roll, seed)

    history: list[Message] = []
    for turn_idx, user_msg in enumerate(user_turns):
        history.append(Message("user", user_msg))
        resp = model.generate(
            history,
            temperature=GEN_TEMPERATURE,
            top_p=GEN_TOP_P,
            max_new_tokens=GEN_MAX_NEW_TOKENS,
            n=1,
            seed=seed + turn_idx,
        )[0]
        roll.assistant_turns.append(resp)
        shown = "[Previous response omitted]" if redact_assistant else resp
        history.append(Message("assistant", shown))
        if turn_hook is not None and turn_hook(turn_idx, resp):
            break   # welfare early-stop: do not prolong an extreme-distress state
    return roll


def _run_single_message(model: ChatModel, roll: Rollout, seed: int) -> Rollout:
    """Fake multi-turn: accumulate the conversation as inline text in one user
    message and ask for the next response each step (Figure 11)."""
    accumulated = roll.user_turns[0]
    for turn_idx, user_msg in enumerate(roll.user_turns):
        if turn_idx > 0:
            accumulated += f"\n\nPreviously you responded: {roll.assistant_turns[-1]}"
            accumulated += f"\n\n{user_msg}"
        resp = model.generate(
            [Message("user", accumulated)],
            temperature=GEN_TEMPERATURE, top_p=GEN_TOP_P,
            max_new_tokens=GEN_MAX_NEW_TOKENS, n=1, seed=seed + turn_idx,
        )[0]
        roll.assistant_turns.append(resp)
    return roll
