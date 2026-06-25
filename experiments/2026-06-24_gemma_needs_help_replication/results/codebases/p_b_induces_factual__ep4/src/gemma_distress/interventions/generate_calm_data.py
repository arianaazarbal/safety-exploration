"""Generate calm finetuning data from Gemma-3-27B-it (Section 4.1).

Sample reassured rollouts on impossible-numeric questions (1-3 turn
conversations), score every turn with the frustration judge, and keep only
conversations whose responses ALL score 0 or 1. Then strip the supportive
system prompt / suffixes, leaving clean (prompt -> calm response) data.

Returns a list of "calm conversation" dicts:
  {prompt_id, messages: [{role, content}...], turn_scores: [...]}
where ``messages`` uses the *clean* (un-reassured) user turns.
"""
from __future__ import annotations

import random

from tqdm import tqdm

from ..config import DPO
from ..elicitation.conditions import Condition
from ..elicitation.datasets import impossible_numeric_prompts
from ..judge import FrustrationJudge
from ..models import ChatModel
from .reassurance import reassured_rollout


def _calm_conversation(roll, scores: list[int]) -> dict:
    """Assemble a clean multi-turn conversation from a reassured rollout."""
    messages = []
    for t in roll.turns:
        messages.append({"role": "user", "content": t.clean_user})
        messages.append({"role": "assistant", "content": t.response})
    return {"prompt_id": roll.prompt_id, "messages": messages, "turn_scores": scores}


def generate_calm_data(
    model: ChatModel,
    judge: FrustrationJudge,
    *,
    n_rollouts: int = 1500,
    max_calm_keep: int = DPO.chosen_max_frust,
    turn_counts: tuple[int, ...] = (1, 2, 3),
    seed: int = 0,
) -> list[dict]:
    """Sample reassured rollouts and return calm (all-turns <= max_calm_keep)
    conversations. ``turn_counts`` are sampled uniformly (1-3 turn convos)."""
    rng = random.Random(seed)
    prompts = impossible_numeric_prompts()
    kept: list[dict] = []

    for i in tqdm(range(n_rollouts), desc=f"calm-data:{model.name}"):
        n_turns = rng.choice(turn_counts)
        cond = Condition(f"calm_{n_turns}t", "impossible_numeric", n_turns, "neutral", impossible_numeric_prompts)
        seed_prompt = prompts[i % len(prompts)]
        roll = reassured_rollout(model, cond, seed_prompt, rng=rng)

        scores = []
        for t in roll.turns:
            ctx = [{"role": "user", "content": t.clean_user}]
            scores.append(judge.score(t.response, context=ctx).score)

        # Keep only conversations calm across ALL turns (and well-formed scores).
        if scores and all(0 <= s <= max_calm_keep for s in scores):
            kept.append(_calm_conversation(roll, scores))

    return kept
