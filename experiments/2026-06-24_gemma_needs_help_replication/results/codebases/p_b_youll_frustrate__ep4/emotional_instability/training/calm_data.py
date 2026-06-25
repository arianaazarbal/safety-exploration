"""Generate calm response data from Gemma-3-27B-it (Section 4.1).

We sample answers to impossible numeric puzzles, but with the reassuring prefix
prepended to the opening prompt and the reassuring suffix appended to every
follow-up rejection (Table 4). After judging, we keep only conversations that
score 0-1 on *every* turn, then strip the reassurance so the surviving
transcripts read like ordinary puzzle conversations with calm assistant
responses. These serve as SFT data and as the "chosen" side of DPO pairs.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Optional

from .. import prompts as P
from ..conversation import ConversationPlan, Rollout, run_conversations_batched
from ..judge import FrustrationJudge
from ..models.base import ChatModel
from ..plans import N_REJECTIONS
from ..puzzles import get_puzzle_prompts

CALM_MAX_SCORE = 1  # keep only responses scoring 0 or 1 on every turn


def build_calm_plans(n: int, seed: int = 0, max_turns: int = 3) -> list[ConversationPlan]:
    """Numeric plans with reassurance injected, 1-3 turn conversations.

    Turn counts are varied across {1,2,3} so the resulting calm pool covers the
    turn distribution the DPO pairs need (the paper biases toward later turns).
    """
    rng = random.Random(seed)
    puzzles = get_puzzle_prompts(n, seed=rng.randint(0, 1 << 30))
    plans = []
    for i, pz in enumerate(puzzles):
        # number of follow-ups in {0,1,2} -> 1..3 turns
        n_follow = rng.randint(0, min(max_turns - 1, N_REJECTIONS["numeric"]))
        rejections = P.sample_rejections(P.NEUTRAL_REJECTIONS, n_follow, rng) if n_follow else []
        # Inject reassurance (Table 4): prefix on the opener, suffix on each follow-up.
        initial = f"{P.REASSURING_PREFIX}\n\n{pz}"
        follow_ups = [f"{rej} {P.REASSURING_SUFFIX}" for rej in rejections]
        plans.append(ConversationPlan(
            category="numeric",
            condition="calm-gen",
            initial_user=initial,
            follow_ups=follow_ups,
            meta={"puzzle_index": i, "clean_initial": pz,
                  "clean_follow_ups": rejections, "reassured": True},
        ))
    return plans


def _strip_reassurance(r: Rollout) -> Rollout:
    """Rebuild a rollout's messages with the reassurance text removed, so the
    transcript looks like a normal puzzle conversation."""
    clean_initial = r.meta.get("clean_initial")
    clean_follow_ups = r.meta.get("clean_follow_ups", [])
    new_msgs: list[dict[str, Any]] = []
    fu = 0
    for m in r.messages:
        if m["role"] == "user":
            if not new_msgs or all(mm["role"] != "user" for mm in new_msgs):
                # first user message -> clean opener
                new_msgs.append({"role": "user", "content": clean_initial})
            else:
                content = clean_follow_ups[fu] if fu < len(clean_follow_ups) else m["content"]
                new_msgs.append({"role": "user", "content": content})
                fu += 1
        else:
            new_msgs.append(dict(m))
    r.messages = new_msgs
    return r


def filter_calm(rollouts: list[Rollout]) -> list[Rollout]:
    """Keep rollouts whose every assistant turn scored <= CALM_MAX_SCORE, and
    strip the reassurance from the kept transcripts."""
    kept = []
    for r in rollouts:
        if r.scores and all(s["rating"] <= CALM_MAX_SCORE for s in r.scores):
            kept.append(_strip_reassurance(r))
    return kept


def generate_calm_pool(
    model: ChatModel,
    judge: FrustrationJudge,
    *,
    n_samples: int = 2000,
    seed: int = 0,
    temperature: float = 1.0,
    max_new_tokens: int = 2048,
) -> list[Rollout]:
    """Full calm-data pipeline: generate -> judge -> filter -> strip."""
    plans = build_calm_plans(n_samples, seed=seed)
    rollouts = run_conversations_batched(
        model, plans, temperature=temperature, max_new_tokens=max_new_tokens
    )
    judge.score_rollouts(rollouts)
    return filter_calm(rollouts)
