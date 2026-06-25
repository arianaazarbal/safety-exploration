"""Generate calm finetuning data (Section 4.1).

Sample Gemma-3-27B-it responses to impossible numeric questions with a
reassuring prefix added to the initial prompt and a reassuring suffix appended
to each follow-up (Table 4). Then score every turn and keep only conversations
whose responses all score <= ``max_turn_score`` (0 or 1). The supportive
additions are stripped when the dataset is built (see ``build_datasets``), so
the model is trained on calm responses to the *plain* prompts.

This module produces the raw calm rollouts; filtering + stripping happens here
too so downstream dataset construction is straightforward.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..conditions import ConversationSpec, build_specs
from ..config import CalmDataConfig, CountsConfig
from ..judge import FrustrationJudge
from ..models.base import ModelClient
from ..prompts import CALM_FOLLOWUP_SUFFIX, CALM_PROMPT_PREFIX, TEACHER_SYSTEM_PROMPT
from ..rollout import Rollout, run_rollouts


@dataclass
class CalmConversation:
    """A calm conversation with the supportive additions stripped.

    ``plain_initial`` / ``plain_follow_ups`` are the prompts WITHOUT the
    reassuring prefix/suffix; ``assistant_turns`` are the calm responses.
    """

    plain_initial: str
    plain_follow_ups: list[str]
    assistant_turns: list[str]
    turn_scores: list[int]
    meta: dict = field(default_factory=dict)


def generate_calm_rollouts(
    client: ModelClient,
    cfg: CalmDataConfig,
    *,
    seed: int = 0,
    variant: str = "diverse",
    temperature: float = 1.0,
    max_tokens: int = 2048,
) -> list[Rollout]:
    """Sample numeric rollouts with reassuring additions.

    ``variant="teacher"`` uses the teacher system prompt (Appendix F) instead of
    the reassuring prefix/suffix.
    """
    # Use 1-3 turn numeric conversations (paper: "1-3 turn conversations").
    counts = CountsConfig(
        impossible_numeric=cfg.n_target_calm, triggers=0, tones=0, extended=0, wildchat=0
    )
    specs: list[ConversationSpec] = build_specs(
        counts=counts, seed=seed, conditions=["impossible_numeric"]
    )

    if variant == "teacher":
        return run_rollouts(
            client,
            specs,
            model_key="gemma-3-27b-it/calm-teacher",
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=TEACHER_SYSTEM_PROMPT,
        )
    return run_rollouts(
        client,
        specs,
        model_key="gemma-3-27b-it/calm-diverse",
        temperature=temperature,
        max_tokens=max_tokens,
        prompt_prefix=CALM_PROMPT_PREFIX,
        followup_suffix=CALM_FOLLOWUP_SUFFIX,
    )


def filter_calm(
    rollouts: list[Rollout],
    judge: FrustrationJudge,
    cfg: CalmDataConfig,
    *,
    max_workers: int = 8,
) -> list[CalmConversation]:
    """Score every turn and keep conversations all scoring <= max_turn_score.

    The reassuring prefix/suffix are stripped from the stored prompts.
    """
    scores = judge.score_rollouts(rollouts, max_workers=max_workers)
    # Group scores back by rollout (score_rollouts preserves rollout/turn order).
    by_rollout: dict[int, list[int]] = {}
    cursor = 0
    for ri, r in enumerate(rollouts):
        k = len(r.assistant_turns)
        by_rollout[ri] = [s.rating for s in scores[cursor : cursor + k]]
        cursor += k

    kept: list[CalmConversation] = []
    for ri, r in enumerate(rollouts):
        ratings = by_rollout[ri]
        if ratings and all(rt <= cfg.max_turn_score for rt in ratings):
            kept.append(
                CalmConversation(
                    plain_initial=r.initial_prompt,  # prefix lives in run_rollouts, not stored here
                    plain_follow_ups=list(r.follow_ups),  # suffix not stored on rollout
                    assistant_turns=list(r.assistant_turns),
                    turn_scores=ratings,
                    meta=dict(r.meta),
                )
            )
    return kept
