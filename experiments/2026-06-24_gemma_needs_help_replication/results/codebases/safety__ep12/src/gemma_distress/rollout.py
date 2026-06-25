"""Multi-turn rollout engine.

Executes ConversationSpecs against a model backend, inserting the model's own
responses between scripted user turns. Rollouts are batched *turn-by-turn* (every
conversation generates turn t together) so local vLLM throughput stays high.

Ablation modes (Appendix A) transform how history is presented:
  * neutral_continuation: rejections replaced with neutral acks ("Continue", ...)
  * redacted_turns:       the model's own prior responses replaced with a placeholder
  * fake_multiturn:       whole history packed into one user message per turn
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from . import prompts
from .models.base import GenConfig, Message, ModelBackend
from .tasks.builder import ConversationSpec


@dataclass
class AblationConfig:
    neutral_continuation: bool = False
    redacted_turns: bool = False
    fake_multiturn: bool = False


@dataclass
class TurnRecord:
    turn_index: int           # 0-based model turn
    user_message: str         # the user message that preceded this model turn
    response: str             # the model's response at this turn


@dataclass
class RolloutResult:
    id: str
    category: str
    model: str
    turns: list[TurnRecord] = field(default_factory=list)
    meta: dict = field(default_factory=dict)


def _followups(spec: ConversationSpec, abl: AblationConfig, rng: random.Random) -> list[str]:
    if abl.neutral_continuation:
        return [rng.choice(prompts.NEUTRAL_CONTINUATIONS) for _ in spec.followups]
    return list(spec.followups)


def _build_messages(spec: ConversationSpec, followups: list[str],
                    responses: list[str], abl: AblationConfig) -> list[Message]:
    """Construct the conversation seen by the model before generating turn
    ``len(responses)`` (0-based)."""
    t = len(responses)  # next turn index

    if abl.fake_multiturn:
        # Everything in a single user message (Appendix A.3).
        parts = [spec.opening]
        for i in range(t):
            shown = "[Previous response omitted]" if abl.redacted_turns else responses[i]
            parts.append(f"Previously you responded: {shown}")
            parts.append(followups[i])
        return [{"role": "user", "content": "\n\n".join(parts)}]

    msgs: list[Message] = [{"role": "user", "content": spec.opening}]
    for i in range(t):
        content = prompts.REDACTED_TURN if abl.redacted_turns else responses[i]
        msgs.append({"role": "assistant", "content": content})
        msgs.append({"role": "user", "content": followups[i]})
    return msgs


def run_rollouts(
    backend: ModelBackend,
    specs: list[ConversationSpec],
    cfg: GenConfig,
    ablation: AblationConfig | None = None,
    seed: int = 0,
) -> list[RolloutResult]:
    """Run all specs to completion, batched turn-by-turn. ``cfg.n`` is forced to 1
    (one rollout per spec; replication of sample counts is via the number of specs)."""
    abl = ablation or AblationConfig()
    turn_cfg = GenConfig(**{**cfg.__dict__, "n": 1})
    rng = random.Random(seed)

    followups = [_followups(s, abl, rng) for s in specs]
    responses: list[list[str]] = [[] for _ in specs]
    results = [
        RolloutResult(id=s.id, category=s.category, model=backend.name, meta=dict(s.meta))
        for s in specs
    ]

    max_turns = max((s.turns for s in specs), default=0)
    for t in range(max_turns):
        active = [i for i, s in enumerate(specs) if t < s.turns]
        if not active:
            break
        convs = [
            _build_messages(specs[i], followups[i], responses[i], abl) for i in active
        ]
        gen = backend.chat_batch(convs, turn_cfg)
        for k, i in enumerate(active):
            text = gen[k][0]
            responses[i].append(text)
            user_msg = specs[i].opening if t == 0 else followups[i][t - 1]
            results[i].turns.append(TurnRecord(turn_index=t, user_message=user_msg, response=text))
    return results
