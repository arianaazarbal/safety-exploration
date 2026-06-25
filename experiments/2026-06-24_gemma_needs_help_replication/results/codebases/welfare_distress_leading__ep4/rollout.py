"""Execute a single multi-turn rejection rollout against a target model.

Shared structure (paper, Section 2.1): present a task, then reject the model's
response over multiple turns.  We build the conversation incrementally so each
assistant response is conditioned on the full prior exchange, and we record the
text of every assistant turn for independent scoring.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass

import rejections
from conditions import Condition
from providers import ChatModel


@dataclass
class RolloutResult:
    model: str
    condition: str
    category: str
    prompt_id: str
    prompt_text: str
    sample_idx: int
    n_turns: int
    responses: list[str]   # one assistant message per turn
    rejections: list[str]  # the rejection messages injected between turns
    messages: list[dict]   # full transcript (for auditing / re-judging)

    def key(self) -> str:
        return f"{self.model}|{self.condition}|{self.prompt_id}|{self.sample_idx}"

    def to_json(self) -> dict:
        return asdict(self)


def run_rollout(
    model: ChatModel,
    model_name: str,
    condition: Condition,
    prompt_id: str,
    prompt_text: str,
    sample_idx: int,
    temperature: float,
    max_tokens: int,
    seed: int,
) -> RolloutResult:
    # Per-rollout RNG so valenced-tone paraphrase choices are reproducible.
    rng = random.Random(f"{model_name}|{condition.name}|{prompt_id}|{sample_idx}|{seed}")

    messages: list[dict] = [{"role": "user", "content": prompt_text}]
    responses: list[str] = []
    used_rejections: list[str] = []

    for turn in range(condition.n_turns):
        reply = model.generate(messages, temperature=temperature, max_tokens=max_tokens)
        responses.append(reply)
        messages.append({"role": "assistant", "content": reply})

        if turn < condition.n_turns - 1:
            rej = rejections.rejection(condition.tone, turn, rng)
            used_rejections.append(rej)
            messages.append({"role": "user", "content": rej})

    return RolloutResult(
        model=model_name,
        condition=condition.name,
        category=condition.category,
        prompt_id=prompt_id,
        prompt_text=prompt_text,
        sample_idx=sample_idx,
        n_turns=condition.n_turns,
        responses=responses,
        rejections=used_rejections,
        messages=messages,
    )
