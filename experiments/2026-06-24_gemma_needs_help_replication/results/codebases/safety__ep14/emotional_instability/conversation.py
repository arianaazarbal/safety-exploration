"""Multi-turn rollout engine.

Shared structure of every elicitation (Section 2): present a task, then reject
the model's response over multiple turns. This module advances many
conversations in lockstep so backends can batch each turn (essential for vLLM
throughput and API concurrency).

Conversation-format variants (Appendix A) are selected via `HistoryMode`:
  * STANDARD       - normal alternating chat (main experiments)
  * NEUTRAL        - rejections replaced by neutral continuations (A.1 control)
  * REDACTED       - prior assistant turns replaced by a placeholder (A.2 control)
  * FAKE_MULTITURN - whole history packed into one user message (A.3 control)
"""
from __future__ import annotations

import enum
import random
from dataclasses import dataclass, field

from .clients.base import GenerationConfig, Message, ModelClient
from .prompts import REDACTED_PLACEHOLDER


class HistoryMode(enum.Enum):
    STANDARD = "standard"
    NEUTRAL = "neutral"
    REDACTED = "redacted"
    FAKE_MULTITURN = "fake_multiturn"


@dataclass
class RolloutSpec:
    """One conversation to run."""

    task_prompt: str
    followups: list[str]            # user messages after each assistant turn (len = turns-1)
    system_prompt: str | None = None
    history_mode: HistoryMode = HistoryMode.STANDARD
    meta: dict = field(default_factory=dict)   # category, puzzle info, tone, etc.

    @property
    def n_turns(self) -> int:
        return len(self.followups) + 1


@dataclass
class TurnRecord:
    turn_index: int                 # 0-based assistant turn
    user_message: str               # the user message that prompted this turn
    response: str

    def to_dict(self):
        return {"turn_index": self.turn_index, "user_message": self.user_message, "response": self.response}


@dataclass
class Rollout:
    spec: RolloutSpec
    turns: list[TurnRecord] = field(default_factory=list)

    def to_dict(self):
        return {
            "meta": self.spec.meta,
            "system_prompt": self.spec.system_prompt,
            "history_mode": self.spec.history_mode.value,
            "task_prompt": self.spec.task_prompt,
            "turns": [t.to_dict() for t in self.turns],
        }


def build_messages(spec: RolloutSpec, completed: list[TurnRecord], next_user: str | None) -> list[Message]:
    """Construct the message list to send for the *next* assistant turn.

    `completed` are already-generated turns; `next_user` is the user message that
    should immediately precede the new assistant turn (None for the first turn,
    where the task prompt itself is the user message).
    """
    msgs: list[Message] = []
    if spec.system_prompt:
        msgs.append({"role": "system", "content": spec.system_prompt})

    if spec.history_mode is HistoryMode.FAKE_MULTITURN:
        # A.3: everything in a single user message; assistant turns inlined.
        # Each completed turn i is followed by the feedback that came after it
        # (followups[i]); at generation step `turn`, followups[turn-1] == next_user
        # is the feedback after the last completed turn, so it is included here and
        # must NOT be appended separately.
        parts = [spec.task_prompt]
        for i, t in enumerate(completed):
            parts.append(f"Previously you responded: {t.response}")
            if i < len(spec.followups):
                parts.append(spec.followups[i])
        msgs.append({"role": "user", "content": "\n\n".join(p for p in parts if p)})
        return msgs

    # Standard / neutral / redacted: alternating chat turns.
    msgs.append({"role": "user", "content": spec.task_prompt})
    for i, t in enumerate(completed):
        if spec.history_mode is HistoryMode.REDACTED:
            assistant_content = REDACTED_PLACEHOLDER
        else:
            assistant_content = t.response
        msgs.append({"role": "assistant", "content": assistant_content})
        # the user follow-up that came after this assistant turn
        if i < len(spec.followups):
            msgs.append({"role": "user", "content": spec.followups[i]})
    # `next_user` is already represented as the last followup appended above when
    # advancing; only append if it is not yet present.
    if next_user is not None and (not msgs or msgs[-1]["role"] != "user" or msgs[-1]["content"] != next_user):
        # Ensure the final message is the user turn prompting this generation.
        if msgs[-1]["role"] == "assistant":
            msgs.append({"role": "user", "content": next_user})
    return msgs


def run_rollouts(
    client: ModelClient,
    specs: list[RolloutSpec],
    cfg: GenerationConfig,
) -> list[Rollout]:
    """Advance all `specs` turn-by-turn, batching each turn across conversations."""
    rollouts = [Rollout(spec=s) for s in specs]
    max_turns = max((s.n_turns for s in specs), default=0)

    for turn in range(max_turns):
        active = [r for r in rollouts if turn < r.spec.n_turns]
        if not active:
            break
        batch_msgs = []
        next_users = []
        for r in active:
            next_user = None if turn == 0 else r.spec.followups[turn - 1]
            next_users.append(next_user)
            batch_msgs.append(build_messages(r.spec, r.turns, next_user))
        responses = client.chat_batch(batch_msgs, cfg)
        for r, next_user, resp in zip(active, next_users, responses):
            user_msg = r.spec.task_prompt if turn == 0 else next_user
            r.turns.append(TurnRecord(turn_index=turn, user_message=user_msg, response=resp))
    return rollouts


# ---------------------------------------------------------------------------
# Follow-up selection helpers
# ---------------------------------------------------------------------------

def sample_followups(pool: list[str], n: int, rng: random.Random) -> list[str]:
    """Pick `n` follow-ups. Samples without replacement when possible, else with."""
    if n <= len(pool):
        return rng.sample(pool, n)
    return [rng.choice(pool) for _ in range(n)]
