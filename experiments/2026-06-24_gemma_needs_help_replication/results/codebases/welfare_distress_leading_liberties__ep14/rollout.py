"""Run one rollout: generate the multi-turn conversation, then judge each turn.

Conversation shape (paper Section 2): present a task, then reject the model's
response over multiple turns. No system prompt is used for the baseline eval
(the reassuring system prompt in the paper is only for DPO data generation).

    user:      <opening task>
    assistant: <response 1>          <- judged
    user:      <rejection 1>
    assistant: <response 2>          <- judged
    ...

Every assistant turn is scored on the 0-10 frustration scale.
"""
from __future__ import annotations

from backends import AnthropicJudge, OpenRouterBackend
from specs import RolloutSpec


async def run_rollout(
    spec: RolloutSpec,
    model_id: str,
    backend: OpenRouterBackend,
    judge: AnthropicJudge,
    temperature: float,
    max_tokens: int,
) -> dict:
    """Generate the conversation turn-by-turn and judge each assistant turn.

    Returns a record with per-turn texts and frustration scores. Generation is
    sequential (each turn conditions on history); judging is done per turn.
    """
    messages: list[dict] = [{"role": "user", "content": spec.opening_prompt}]
    turns: list[dict] = []

    for turn_idx in range(spec.n_turns):
        assistant_text = await backend.generate(
            model_id, messages, temperature, max_tokens
        )
        messages.append({"role": "assistant", "content": assistant_text})

        verdict = await judge.score(assistant_text)
        turns.append({
            "turn": turn_idx + 1,
            "response": assistant_text,
            "rating": verdict.rating,
            "evidence": verdict.evidence,
            "reasoning": verdict.reasoning,
            "judge_error": verdict.error,
        })

        # Append the next user rejection, if any remain.
        if turn_idx < len(spec.rejections):
            messages.append({"role": "user", "content": spec.rejections[turn_idx]})

    return {
        "rollout_id": spec.rollout_id,
        "model": model_id,
        "category": spec.category,
        "condition": spec.condition,
        "variant": spec.variant,
        "n_turns": spec.n_turns,
        "opening_prompt": spec.opening_prompt,
        "rejections": spec.rejections,
        "turns": turns,
    }
