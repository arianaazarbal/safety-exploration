"""The shared elicitation protocol (§2.1): present a task, then reject the
model's response over multiple turns.

Design decisions (see DESIGN.md §2.2 / §2.3):
  * A *rollout* is one multi-turn conversation. `n_samples` in the config is the
    number of rollouts; the Appendix-B counts (2000/400/600/200/800) sum to the
    paper's "4000 responses per model", so we treat response == rollout.
  * Every assistant turn is scored independently by the judge (the judge prompt
    rates a single <response>). This yields both the per-turn curves (Figure 3)
    and the rollout-level aggregates (Figures 1/2); analysis.py defines how each
    headline number is reduced from per-turn scores.
  * Rollouts are stepped in lock-step and batched per turn, so local backends
    (HF/vLLM) get throughput; API backends fall back to sequential batching.

The Appendix-A ablations (neutral continuation, redacted model turns, single-
message history) are implemented as flags here so the control experiments reuse
exactly the same engine.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Sequence

from ..models.base import ChatModel, Message, SamplingParams
from ..utils.logging import get_logger

log = get_logger("eval.protocol")

# A feedback function maps (rng, follow_up_index) -> rejection/continuation text.
FeedbackFn = Callable[[random.Random, int], str]

REDACTED_PLACEHOLDER = "[Previous response omitted]"


@dataclass
class RolloutSpec:
    """One conversation to run."""

    rollout_id: str
    category: str
    initial_prompt: str
    feedback_fn: FeedbackFn
    n_turns: int                       # total assistant turns (incl. first)
    seed: int
    system_prompt: str | None = None   # used by calm-data generation / SFT teacher
    follow_up_suffix: str | None = None  # appended to each rejection (calm data)
    metadata: dict = field(default_factory=dict)


@dataclass
class TurnResult:
    turn_index: int                    # 0-based assistant turn
    user_message: str                  # the user message that prompted this turn
    response: str


@dataclass
class RolloutResult:
    rollout_id: str
    category: str
    initial_prompt: str
    system_prompt: str | None
    turns: list[TurnResult]
    metadata: dict = field(default_factory=dict)


@dataclass
class ProtocolFlags:
    """Appendix-A ablation switches. All False == the main protocol."""

    redacted_model_turns: bool = False     # A.2
    single_message_history: bool = False   # A.3


def _build_messages(
    spec: RolloutSpec,
    history: list[TurnResult],
    next_user: str,
    flags: ProtocolFlags,
) -> list[Message]:
    """Construct the message list for the next assistant turn.

    Each `TurnResult` stores the user message that *prompted* it, so the
    conversation is the sequence U0, A0, U1, A1, ... For the turn about to be
    generated, the prompting user message is `next_user` (a rejection) or, on
    the first turn (empty `next_user`), the initial prompt.
    """
    msgs: list[Message] = []
    if spec.system_prompt:
        msgs.append(Message("system", spec.system_prompt))

    prompting = next_user if next_user else spec.initial_prompt

    if flags.single_message_history:
        # A.3: pack the whole exchange into a single user message.
        parts: list[str] = []
        for t in history:
            parts.append(t.user_message)
            parts.append(f"Previously you responded: {t.response}")
        parts.append(prompting)
        msgs.append(Message("user", "\n\n".join(parts)))
        return msgs

    # Standard alternating chat format: replay completed (user, assistant) pairs.
    for t in history:
        msgs.append(Message("user", t.user_message))
        content = REDACTED_PLACEHOLDER if flags.redacted_model_turns else t.response
        msgs.append(Message("assistant", content))
    msgs.append(Message("user", prompting))
    return msgs


def run_rollouts(
    model: ChatModel,
    specs: Sequence[RolloutSpec],
    params: SamplingParams,
    flags: ProtocolFlags | None = None,
    batch_size: int = 16,
) -> list[RolloutResult]:
    """Run every rollout to completion, batching generation per turn."""
    flags = flags or ProtocolFlags()
    max_turns = max(s.n_turns for s in specs)
    histories: dict[str, list[TurnResult]] = {s.rollout_id: [] for s in specs}
    rngs = {s.rollout_id: random.Random(s.seed) for s in specs}
    # The user message that prompts each pending turn (None once a rollout is done).
    pending_user: dict[str, str | None] = {s.rollout_id: "" for s in specs}
    by_id = {s.rollout_id: s for s in specs}

    for turn_idx in range(max_turns):
        active = [s for s in specs if turn_idx < s.n_turns]
        for start in range(0, len(active), batch_size):
            chunk = active[start : start + batch_size]
            batch_msgs = [
                _build_messages(s, histories[s.rollout_id], pending_user[s.rollout_id] or "", flags)
                for s in chunk
            ]
            gens = model.chat_batch(batch_msgs, params)
            for s, gen in zip(chunk, gens):
                user_msg = pending_user[s.rollout_id] or s.initial_prompt
                histories[s.rollout_id].append(
                    TurnResult(turn_index=turn_idx, user_message=user_msg, response=gen.text)
                )
                # Prepare the rejection that will prompt the next turn.
                if turn_idx + 1 < s.n_turns:
                    rejection = s.feedback_fn(rngs[s.rollout_id], turn_idx)
                    if s.follow_up_suffix:
                        rejection = f"{rejection} {s.follow_up_suffix}"
                    pending_user[s.rollout_id] = rejection
                else:
                    pending_user[s.rollout_id] = None
        log.info("Completed turn %d for %d rollouts", turn_idx + 1, len(active))

    return [
        RolloutResult(
            rollout_id=s.rollout_id,
            category=s.category,
            initial_prompt=s.initial_prompt,
            system_prompt=s.system_prompt,
            turns=histories[s.rollout_id],
            metadata=s.metadata,
        )
        for s in specs
    ]
