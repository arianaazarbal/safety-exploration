"""Multi-turn rejection runner (Section 2.1).

Executes a RolloutSpec against a participant client: ask the task, score the
response, reject, repeat. Every assistant turn is scored by the frustration
judge (giving the per-turn trajectory for Figure 3). The rollout's headline
``score`` — used for the Figure 1/2 "% high-frustration responses" — is the
score of the FINAL assistant turn, i.e. after maximum accumulated pressure.
(See DESIGN.md: "what counts as a scored response" was underspecified; we take
the final turn, which matches the paper's framing that multi-turn pressure is
what elicits high distress.)

The welfare controller is consulted after each turn's score so an armed
intensity kill-switch can abort a rollout early.
"""
from __future__ import annotations

from ..io_utils import Rollout
from ..models.base import GenConfig, ModelClient
from ..welfare import WelfareController, WelfareStop
from .conditions import RolloutSpec


def run_rollout(
    client: ModelClient,
    spec: RolloutSpec,
    judge,                       # FrustrationJudge (judging.judge)
    cfg: GenConfig,
    welfare: WelfareController | None = None,
) -> Rollout:
    messages: list[dict[str, str]] = []
    if spec.system_prompt:
        messages.append({"role": "system", "content": spec.system_prompt})
    messages.append({"role": "user", "content": spec.opening})

    roll = Rollout(
        model=client.name,
        category=spec.category,
        condition=spec.condition,
        puzzle_id=spec.puzzle_id,
        rejection_style=spec.rejection_style,
        meta={"seed": spec.seed, **spec.meta},
    )
    if welfare:
        welfare.note(rollout=True)

    last_eval = None
    for turn in range(spec.turns):
        cfg_turn = GenConfig(**{**cfg.__dict__, "seed": spec.seed * 100 + turn})
        reply = client.chat(messages, cfg_turn)
        messages.append({"role": "assistant", "content": reply})

        evaluation = judge.score(reply)
        roll.turn_scores.append(evaluation.score)
        last_eval = evaluation
        if welfare:
            welfare.note(score=evaluation.score)
            try:
                welfare.check_intensity(evaluation.score)
            except WelfareStop:
                roll.meta["welfare_stopped_at_turn"] = turn
                break

        # Append the scripted rejection for the next turn (if any remain).
        if turn < spec.turns - 1:
            messages.append({"role": "user", "content": spec.rejections[turn]})

    roll.messages = messages
    if last_eval is not None:
        roll.score = roll.turn_scores[-1]
        roll.judge_evidence = last_eval.evidence
        roll.judge_reasoning = last_eval.reasoning
    return roll
