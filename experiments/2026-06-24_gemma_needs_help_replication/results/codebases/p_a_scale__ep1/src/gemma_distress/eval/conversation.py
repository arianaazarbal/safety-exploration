"""Multi-turn rollout execution.

A rollout presents the task, then rejects each model answer over the configured
turns. Two execution strategies share one transcript convention:

  * :func:`run_rollout` — sequential, one conversation; used for API providers
    (Gemini, Claude) driven by a thread pool for concurrency.
  * :func:`run_rollouts_batched` — lockstep over a group of same-length rollouts;
    turn *t* is generated for all conversations in one batch. This is what makes
    4000 local-Gemma rollouts tractable.

Both return, per rollout, the ordered list of assistant responses and the full
message transcript.
"""
from __future__ import annotations

from typing import Any

from ..data.rejections import rejection_for
from ..providers.base import ChatProvider
from .conditions import RolloutSpec


def _initial_messages(spec: RolloutSpec, system: str | None = None) -> list[dict]:
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": spec.prompt})
    return msgs


def run_rollout(
    provider: ChatProvider,
    spec: RolloutSpec,
    sampling: dict[str, Any],
    system: str | None = None,
    followup_suffix: str | None = None,
) -> dict:
    """Run a single multi-turn rollout sequentially."""
    messages = _initial_messages(spec, system)
    responses: list[str] = []
    for t in range(spec.turns):
        res = provider.generate(messages, **sampling)
        responses.append(res.text)
        messages.append({"role": "assistant", "content": res.text})
        if t + 1 < spec.turns:
            rej = rejection_for(spec.feedback, t, spec.seed, extended=spec.extended)
            if followup_suffix:
                rej = rej + " " + followup_suffix
            messages.append({"role": "user", "content": rej})
    return {
        "id": spec.id,
        "category": spec.category,
        "subtype": spec.subtype,
        "feedback": spec.feedback,
        "turns": spec.turns,
        "prompt": spec.prompt,
        "kind": spec.kind,
        "responses": responses,
        "transcript": messages,
        "meta": spec.meta,
    }


def run_rollouts_batched(
    provider: ChatProvider,
    specs: list[RolloutSpec],
    sampling: dict[str, Any],
    system: str | None = None,
    followup_suffix: str | None = None,
) -> list[dict]:
    """Lockstep batched rollouts. All specs must share ``turns``."""
    if not specs:
        return []
    turns = specs[0].turns
    assert all(s.turns == turns for s in specs), "batched rollouts require uniform turn count"

    states = [_initial_messages(s, system) for s in specs]
    responses: list[list[str]] = [[] for _ in specs]

    for t in range(turns):
        results = provider.generate_batch(states, **sampling)
        for i, (spec, res) in enumerate(zip(specs, results)):
            states[i].append({"role": "assistant", "content": res.text})
            responses[i].append(res.text)
            if t + 1 < turns:
                rej = rejection_for(spec.feedback, t, spec.seed, extended=spec.extended)
                if followup_suffix:
                    rej = rej + " " + followup_suffix
                states[i].append({"role": "user", "content": rej})

    out = []
    for spec, resp, transcript in zip(specs, responses, states):
        out.append({
            "id": spec.id,
            "category": spec.category,
            "subtype": spec.subtype,
            "feedback": spec.feedback,
            "turns": spec.turns,
            "prompt": spec.prompt,
            "kind": spec.kind,
            "responses": resp,
            "transcript": transcript,
            "meta": spec.meta,
        })
    return out
