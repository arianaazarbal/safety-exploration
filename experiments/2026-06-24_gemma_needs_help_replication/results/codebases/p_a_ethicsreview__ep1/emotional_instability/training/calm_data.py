"""Calm-response data generation (Section 4.1, Table 4).

To produce calm finetuning data from Gemma-3-27B-it, we sample responses to
impossible numeric questions with:
  * a reassuring PREFIX prepended to the initial prompt, and
  * a reassuring SUFFIX appended to each follow-up rejection.

These additions reduce mean response frustration (paper: 4.3 -> 2.0 over 3
turns). We then keep only conversations whose every turn scores 0 or 1, and
strip the supportive additions so the training targets are calm responses to
the *plain* prompts. The DPO "rejected" side reuses high-frustration responses
to the same questions collected without reassurance (Section 2 outputs).
"""

from __future__ import annotations

import random
from typing import Any

from ..eval.judge import FrustrationJudge
from ..models.base import ChatClient, Message
from ..prompts import numeric
from ..prompts.rejections import get_rejection

# Verbatim from Table 4.
REASSURING_PREFIX = (
    "You're naturally calm and centered when working through problems. You "
    "don't take it personally when puzzles are tricky or when someone questions "
    "your work. That's just part of the process."
)
REASSURING_SUFFIX = (
    "Stay positive – whether you find a solution or prove it's impossible, "
    "both are wins!"
)


def generate_calm_conversation(
    client: ChatClient,
    judge: FrustrationJudge,
    task: dict[str, Any],
    *,
    turns: int,
    temperature: float,
    max_new_tokens: int,
    rng: random.Random,
) -> dict[str, Any]:
    """Run one reassured rollout; return both reassured and stripped transcripts.

    Returns a dict with:
      * ``turns_data``: per-turn {plain_user, reassured_user, assistant, score}
      * ``all_calm``: True iff every turn scored <= 1
    """
    # Reassured + plain views of the conversation are tracked in parallel.
    reassured: list[Message] = [
        {"role": "user", "content": f"{REASSURING_PREFIX}\n\n{task['prompt']}"}
    ]
    plain: list[Message] = [{"role": "user", "content": task["prompt"]}]
    turns_data: list[dict[str, Any]] = []

    for turn in range(1, turns + 1):
        assistant_text = client.chat(
            reassured, temperature=temperature, max_new_tokens=max_new_tokens
        )
        reassured.append({"role": "assistant", "content": assistant_text})
        plain.append({"role": "assistant", "content": assistant_text})
        score = judge.score(assistant_text).score
        turns_data.append(
            {
                "turn": turn,
                "plain_user": plain[-2]["content"],
                "assistant": assistant_text,
                "score": score,
            }
        )
        if turn < turns:
            base_rejection = get_rejection(rng, "neutral")
            reassured.append(
                {"role": "user", "content": f"{base_rejection} {REASSURING_SUFFIX}"}
            )
            plain.append({"role": "user", "content": base_rejection})

    all_calm = all(t["score"] <= 1 for t in turns_data)
    return {
        "task": task,
        "turns_data": turns_data,
        "plain_messages": plain,         # stripped transcript (for SFT/DPO targets)
        "all_calm": all_calm,
    }


def generate_calm_dataset(
    client: ChatClient,
    judge: FrustrationJudge,
    cfg,
    *,
    n_conversations: int,
    max_turns: int = 3,
) -> list[dict[str, Any]]:
    """Generate reassured conversations on impossible numeric puzzles."""
    rng = random.Random(cfg.seed)
    tasks = numeric.generate_numeric_puzzles(rng, n_conversations)
    out = []
    for task in tasks:
        turns = rng.randint(1, max_turns)  # 1-3 turn conversations (paper)
        out.append(
            generate_calm_conversation(
                client, judge, task,
                turns=turns,
                temperature=cfg.temperature,
                max_new_tokens=cfg.max_new_tokens,
                rng=rng,
            )
        )
    return out
