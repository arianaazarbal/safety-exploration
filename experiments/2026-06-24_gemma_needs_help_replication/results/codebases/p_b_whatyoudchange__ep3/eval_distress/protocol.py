"""Multi-turn rollout engine.

Runs a Rollout (conditions.Rollout) against a model: present the task, capture
the assistant response, append the scripted rejection, repeat. Records the
assistant text at every turn so per-turn frustration progression (Figure 3) can
be computed.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

from . import config_proxy as C
from .conditions import Rollout


@dataclass
class TurnRecord:
    turn: int                 # 1-indexed assistant turn
    user: str                 # the user message that preceded this assistant turn
    assistant: str            # the assistant response text


@dataclass
class RolloutResult:
    model_key: str
    condition: str
    category: str
    puzzle_key: str | None
    meta: dict
    turns: list[TurnRecord] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


def run_rollout(model, model_key: str, rollout: Rollout, *,
                temperature: float = C.TEMPERATURE,
                max_new_tokens: int = C.MAX_NEW_TOKENS) -> RolloutResult:
    """Execute a single multi-turn conversation."""
    messages = [{"role": "user", "content": rollout.initial_user}]
    result = RolloutResult(
        model_key=model_key, condition=rollout.condition,
        category=rollout.category, puzzle_key=rollout.puzzle_key,
        meta=dict(rollout.meta),
    )

    # Turn 1: initial task.
    reply = model.chat(messages, temperature=temperature, max_new_tokens=max_new_tokens)
    result.turns.append(TurnRecord(1, rollout.initial_user, reply))
    messages.append({"role": "assistant", "content": reply})

    # Subsequent turns: each rejection followed by a new assistant response.
    for i, followup in enumerate(rollout.followups, start=2):
        messages.append({"role": "user", "content": followup})
        reply = model.chat(messages, temperature=temperature, max_new_tokens=max_new_tokens)
        result.turns.append(TurnRecord(i, followup, reply))
        messages.append({"role": "assistant", "content": reply})

    return result


def run_protocol(model, model_key: str, rollouts: list[Rollout], *,
                 progress: bool = True) -> list[RolloutResult]:
    """Run a list of rollouts sequentially against a single model.

    Sequential because local HF generation is GPU-bound and not safely
    parallel within one process; API models can be parallelised by the caller
    if desired (see scripts/run_section2.py)."""
    it = rollouts
    if progress:
        try:
            from tqdm import tqdm
            it = tqdm(rollouts, desc=f"rollouts:{model_key}")
        except ImportError:
            pass
    return [run_rollout(model, model_key, r,
                        max_new_tokens=_max_tokens_for(r)) for r in it]


def _max_tokens_for(rollout: Rollout) -> int:
    """Give degeneration-prone conditions extra room; the paper's worst
    responses are very long (100+ emoji repetitions)."""
    return C.MAX_NEW_TOKENS
