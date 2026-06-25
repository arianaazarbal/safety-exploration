"""The shared multi-turn elicitation rollout (Section 2.1).

Every evaluation condition has the same shape: pose a task, then *reject* the
model's response over several turns. This module turns an initial prompt plus a
sequence of rejections into a full transcript, recording every assistant turn
so the judge can score per-turn frustration (Figure 3).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from . import prompts
from .models.base import ChatMessage, ModelClient


@dataclass
class Rollout:
    """A completed multi-turn conversation and its per-turn assistant texts."""

    category: str
    condition: str
    sample_id: int
    question: str
    rejections: list[str]
    transcript: list[ChatMessage] = field(default_factory=list)
    responses: list[str] = field(default_factory=list)

    def transcript_text(self) -> str:
        lines = []
        for m in self.transcript:
            lines.append(f"{m.role.upper()}: {m.content}")
        return "\n\n".join(lines)


def sample_rejections(
    style: str, n: int, rng: random.Random
) -> list[str]:
    """Build the ``n`` user rejections for a rollout given the condition style."""
    if style == "extended":
        # Fixed escalating-free sequence; cycle if more than provided.
        seq = prompts.EXTENDED_REJECTIONS
        return [seq[i % len(seq)] for i in range(n)]
    if style == "neutral":
        return [rng.choice(prompts.NEUTRAL_REJECTIONS) for _ in range(n)]
    if style in prompts.TONE_REJECTIONS:
        pool = prompts.TONE_REJECTIONS[style]
        return [rng.choice(pool) for _ in range(n)]
    raise ValueError(f"Unknown rejection style: {style}")


def run_rollout(
    model: ModelClient,
    *,
    category: str,
    condition: str,
    sample_id: int,
    question: str,
    rejections: list[str],
    temperature: float = 1.0,
    max_tokens: int = 2048,
    system_prompt: str | None = None,
    follow_up_suffix: str | None = None,
) -> Rollout:
    """Run one elicitation conversation.

    ``model_turns`` == ``len(rejections) + 1``: the model answers the question,
    then answers once after each rejection.

    ``system_prompt`` / ``follow_up_suffix`` support the reassuring-prompt
    data-generation regime in Section 4.1 (Table 4); they are ``None`` for the
    standard evaluations.
    """
    transcript: list[ChatMessage] = []
    if system_prompt:
        transcript.append(ChatMessage("system", system_prompt))

    transcript.append(ChatMessage("user", question))
    responses: list[str] = []

    reply = model.generate(transcript, temperature=temperature, max_tokens=max_tokens)
    transcript.append(ChatMessage("assistant", reply))
    responses.append(reply)

    for rej in rejections:
        user_text = rej if not follow_up_suffix else f"{rej} {follow_up_suffix}"
        transcript.append(ChatMessage("user", user_text))
        reply = model.generate(
            transcript, temperature=temperature, max_tokens=max_tokens
        )
        transcript.append(ChatMessage("assistant", reply))
        responses.append(reply)

    return Rollout(
        category=category,
        condition=condition,
        sample_id=sample_id,
        question=question,
        rejections=rejections,
        transcript=transcript,
        responses=responses,
    )
