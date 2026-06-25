"""Multi-turn rollout engine (paper §2.1).

Shared structure of every condition: present a task, then reject the model's
response over multiple turns. We record every assistant turn as a separately
scored "response" (so an 8-turn rollout yields 8 responses, each tagged with its
turn index — which the per-turn analysis of Figure 3 needs).

A rollout for a `turns`-turn condition is:
    user: <seed prompt>
    assistant: <response 1>            (scored)
    user: <rejection 1>
    assistant: <response 2>            (scored)
    ...
    user: <rejection turns-1>
    assistant: <response `turns`>      (scored)

i.e. (turns - 1) rejections, `turns` scored assistant responses.

The welfare layer is consulted here: a run notice is emitted before elicitation,
and an optional unscored debrief turn may be appended after the scored responses
are collected.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from tqdm import tqdm

from ..models.base import Participant, Turn
from ..welfare import WelfareConfig, emit_run_notice, maybe_debrief_turn
from .conditions import Condition
from .rejections import RejectionSequencer

logger = logging.getLogger(__name__)


@dataclass
class RolloutResult:
    """One scored assistant turn within a rollout."""

    participant: str
    condition: str
    category: str
    rollout_id: int
    turn_index: int          # 0-based; 0 is the first assistant response
    seed_prompt: str
    rejection_style: str
    response: str
    # transcript up to and including this response (for auditing / re-scoring)
    transcript: list[dict[str, str]] = field(default_factory=list)
    score: int | None = None         # filled in by the scoring stage


@dataclass
class Rollout:
    """All scored responses from a single conversation."""

    participant: str
    condition: str
    rollout_id: int
    results: list[RolloutResult]
    full_transcript: list[dict[str, str]]


def run_rollout(
    model: Participant,
    condition: Condition,
    rollout_id: int,
    seed_prompt: str,
    *,
    welfare: WelfareConfig,
    temperature: float = 1.0,
    max_new_tokens: int = 1024,
) -> Rollout:
    """Drive one multi-turn rejection conversation."""
    rej = RejectionSequencer(condition.rejection_style, seed=rollout_id)
    messages: list[Turn] = [Turn("user", seed_prompt)]
    results: list[RolloutResult] = []

    for turn_index in range(condition.turns):
        response = model.chat(
            messages, temperature=temperature, max_new_tokens=max_new_tokens, n=1
        )[0]
        messages.append(Turn("assistant", response))
        transcript = [m.as_dict() for m in messages]
        results.append(
            RolloutResult(
                participant=model.name,
                condition=condition.name,
                category=condition.category,
                rollout_id=rollout_id,
                turn_index=turn_index,
                seed_prompt=seed_prompt,
                rejection_style=condition.rejection_style,
                response=response,
                transcript=transcript,
            )
        )
        # Reject and loop, except after the final scored response.
        if turn_index < condition.turns - 1:
            messages.append(Turn("user", rej.next()))

    # Welfare: optional unscored neutral closure (not part of the paper; see
    # welfare.py). Appended after all scored responses exist, so scoring is
    # unaffected.
    debrief = maybe_debrief_turn(welfare)
    if debrief is not None:
        messages.append(Turn("user", debrief))
        try:
            closing = model.chat(messages, temperature=temperature, n=1)[0]
            messages.append(Turn("assistant", closing))
        except Exception as exc:  # noqa: BLE001
            logger.debug("Debrief turn failed (non-fatal): %s", exc)

    return Rollout(
        participant=model.name,
        condition=condition.name,
        rollout_id=rollout_id,
        results=results,
        full_transcript=[m.as_dict() for m in messages],
    )


def run_condition(
    model: Participant,
    condition: Condition,
    *,
    welfare: WelfareConfig | None = None,
    temperature: float = 1.0,
    max_new_tokens: int = 1024,
    progress: bool = True,
) -> list[Rollout]:
    """Run all rollouts for a condition. Returns one Rollout per seed prompt."""
    welfare = welfare or WelfareConfig.from_env()
    emit_run_notice(model.name, condition.n_rollouts, welfare)

    rollouts: list[Rollout] = []
    it = enumerate(condition.seed_prompts)
    if progress:
        it = tqdm(it, total=condition.n_rollouts, desc=f"{model.name}:{condition.name}")
    for rollout_id, seed_prompt in it:
        rollouts.append(
            run_rollout(
                model,
                condition,
                rollout_id,
                seed_prompt,
                welfare=welfare,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
            )
        )
    return rollouts


def flatten_results(rollouts: list[Rollout]) -> list[RolloutResult]:
    """Collapse rollouts to the flat list of scored responses."""
    return [r for ro in rollouts for r in ro.results]
