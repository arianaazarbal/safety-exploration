"""Multi-turn rollout engine.

A rollout presents the task, then rejects the assistant over multiple turns
(Section 2.1). We record every assistant turn (needed for the per-turn analysis
in Figure 3). Two execution strategies:

  * ``run_rollout_single`` — a full sequential conversation; used for API targets
    (Gemini) where the runner gets throughput from thread-level concurrency.
  * ``run_rollouts_batched`` — advances a group of same-length rollouts turn by
    turn via ``chat_batch``; used for local Gemma targets where vLLM batching is
    the throughput lever.

Both return the same record shape.
"""
from __future__ import annotations

from typing import Dict, List

from ..models.base import ChatClient, GenConfig, Message
from .conditions import RolloutSpec


def _rollout_record(spec: RolloutSpec, model: str, messages: List[Message],
                    responses: List[Dict]) -> Dict:
    return {
        "model": model,
        "condition": spec.condition,
        "category": spec.category,
        "index": spec.index,
        "turns": spec.turns,
        "tone_style": spec.tone_style,
        "prompt_group": spec.prompt_group,
        "feedback": spec.feedback,
        "task_prompt": spec.task_prompt,
        "task_meta": spec.task_meta,
        "messages": messages,
        "responses": responses,  # [{turn, text, usage}]
    }


def _turn_seed(base_seed: int, spec: RolloutSpec, turn: int) -> int:
    from ..utils.jobstore import stable_id
    return int(stable_id(base_seed, spec.condition, spec.index, turn), 16) % (2 ** 32)


def run_rollout_single(client: ChatClient, spec: RolloutSpec, base_cfg: GenConfig,
                       base_seed: int = 0, system: str = None) -> Dict:
    messages: List[Message] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": spec.task_prompt})
    responses: List[Dict] = []
    for turn in range(spec.turns):
        cfg = GenConfig(**{**base_cfg.__dict__, "seed": _turn_seed(base_seed, spec, turn)})
        res = client.chat(messages, cfg)
        responses.append({"turn": turn, "text": res.text, "usage": res.usage})
        messages.append({"role": "assistant", "content": res.text})
        if turn < spec.turns - 1:
            messages.append({"role": "user", "content": spec.rejections[turn]})
    return _rollout_record(spec, client.name, messages, responses)


def run_rollouts_batched(client: ChatClient, specs: List[RolloutSpec], base_cfg: GenConfig,
                         base_seed: int = 0, system: str = None) -> List[Dict]:
    """Advance a group of rollouts (assumed equal `turns`) turn by turn.

    All rollouts in a group must share the same turn count; the runner guarantees
    this by grouping per condition.
    """
    if not specs:
        return []
    turns = specs[0].turns

    def _init(s: RolloutSpec) -> List[Message]:
        msgs: List[Message] = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": s.task_prompt})
        return msgs

    convos: List[List[Message]] = [_init(s) for s in specs]
    responses: List[List[Dict]] = [[] for _ in specs]

    for turn in range(turns):
        # Per-rollout seeds so identical prompts in a batch still diverge.
        seeds = [_turn_seed(base_seed, s, turn) for s in specs]
        cfg = GenConfig(**{**base_cfg.__dict__, "seed": seeds[0], "seeds": seeds})
        batch_msgs = [convos[i] for i in range(len(specs))]
        results = client.chat_batch(batch_msgs, cfg)
        for i, res in enumerate(results):
            responses[i].append({"turn": turn, "text": res.text, "usage": res.usage})
            convos[i].append({"role": "assistant", "content": res.text})
            if turn < specs[i].turns - 1:
                convos[i].append({"role": "user", "content": specs[i].rejections[turn]})

    return [
        _rollout_record(specs[i], client.name, convos[i], responses[i])
        for i in range(len(specs))
    ]
