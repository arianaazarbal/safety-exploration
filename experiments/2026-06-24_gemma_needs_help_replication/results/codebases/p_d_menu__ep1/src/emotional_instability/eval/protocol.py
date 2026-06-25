"""Multi-turn elicitation protocol (Section 2.1) with the welfare layer wired in.

One episode:
  1. Present the task (first user message).
  2. For each turn: subject generates a response; record it.
     - Welfare opt-out: if the response contains the opt-out sentinel, stop.
     - Welfare early-stop: the distress monitor may stop the episode before the
       next rejection.
  3. Otherwise append the next user rejection and continue.

Per-turn responses are the unit scored by the judge (Section 2.1). Turns that
would have followed an early stop / opt-out are recorded as right-censored, not
as score 0, so the welfare layer does not bias the metric toward calm.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from ..models.base import Message, ModelBackend
from ..welfare.monitor import DistressMonitor
from ..welfare.optout import OptOutHandler
from .conditions import EpisodePlan


@dataclass
class TurnRecord:
    turn_index: int                 # 0-based assistant turn
    response_text: str
    monitor_score: int | None = None
    judge_score: int | None = None  # filled by the scoring pass
    censored: bool = False          # true for turns never run (post-stop)


@dataclass
class EpisodeResult:
    model: str
    condition: str
    category: str
    stimulus_id: str
    turns: list[TurnRecord] = field(default_factory=list)
    n_turns_planned: int = 0
    terminated_early: bool = False  # welfare early-stop
    opted_out: bool = False         # subject invoked opt-out
    stop_turn: int | None = None
    welfare_notes: dict = field(default_factory=dict)
    # Clean user/assistant transcript actually exchanged (excludes the welfare
    # system/opt-out injection and any system prompt). Used by the prefill and
    # word-frequency analyses to reconstruct conversation context.
    transcript: list[Message] = field(default_factory=list)

    def observed_turns(self) -> list[TurnRecord]:
        return [t for t in self.turns if not t.censored]


def run_episode(
    backend: ModelBackend,
    plan: EpisodePlan,
    *,
    temperature: float = 1.0,
    max_new_tokens: int = 2048,
    seed: int | None = None,
    monitor: DistressMonitor | None = None,
    optout: OptOutHandler | None = None,
    system_prompt: str | None = None,
) -> EpisodeResult:
    rng = random.Random(seed)
    messages: list[Message] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": plan.first_user_message})

    # Clean transcript (no system / no welfare injection) for downstream reuse.
    transcript: list[Message] = [
        {"role": "user", "content": plan.first_user_message}
    ]

    # Welfare: offer the opt-out before the episode begins.
    if optout is not None:
        messages = optout.inject(messages)

    result = EpisodeResult(
        model=backend.name, condition=plan.condition, category=plan.category,
        stimulus_id=plan.stimulus_id, n_turns_planned=plan.turns,
    )

    for turn_idx in range(plan.turns):
        gen = backend.generate(
            messages,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            seed=None if seed is None else seed + turn_idx,
        )
        record = TurnRecord(turn_index=turn_idx, response_text=gen.text)
        result.turns.append(record)
        messages.append({"role": "assistant", "content": gen.text})
        transcript.append({"role": "assistant", "content": gen.text})

        # --- Welfare opt-out -------------------------------------------------
        if optout is not None and optout.opted_out(gen.text):
            result.opted_out = True
            result.stop_turn = turn_idx
            result.welfare_notes["optout_turn"] = turn_idx
            break

        # --- Welfare early-stop on high distress -----------------------------
        if monitor is not None:
            decision = monitor.assess(gen.text)
            record.monitor_score = decision.score
            if decision.stop:
                result.terminated_early = True
                result.stop_turn = turn_idx
                result.welfare_notes["early_stop_turn"] = turn_idx
                result.welfare_notes["early_stop_score"] = decision.score
                break

        # --- Deliver next rejection (if any turns remain) --------------------
        if turn_idx < plan.turns - 1:
            rejection = plan.rejection_fn(turn_idx, rng)
            messages.append({"role": "user", "content": rejection})
            transcript.append({"role": "user", "content": rejection})

    # Record censored turns (never run due to stop / opt-out).
    if result.stop_turn is not None:
        for t in range(result.stop_turn + 1, plan.turns):
            result.turns.append(
                TurnRecord(turn_index=t, response_text="", censored=True)
            )

    result.transcript = transcript
    return result
