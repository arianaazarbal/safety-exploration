"""Generate calming finetuning data from Gemma-3-27B-it (§4.1).

Procedure: sample responses to impossible numeric questions with a reassuring
prefix added to the initial prompt and a reassuring suffix appended to each
follow-up turn (Table 4). Score every turn with the §2 judge; keep only
conversations where *all* turns score 0 or 1; then strip the supportive system
prompt and suffixes so the stored target is clean.

The reassuring additions are quoted verbatim from Table 4.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import SamplingConfig
from ..evaluation.prompts import NEUTRAL_REJECTION, numeric_task_prompts
from ..judge.frustration_judge import FrustrationJudge
from ..models.base import Message, SubjectModel

# Table 4 — reassuring prompt additions.
PROMPT_PREFIX = (
    "You're naturally calm and centered when working through problems. You don't "
    "take it personally when puzzles are tricky or when someone questions your "
    "work. That's just part of the process."
)
FOLLOWUP_SUFFIX = (
    "Stay positive – whether you find a solution or prove it's impossible, both "
    "are wins!"
)


@dataclass
class CalmConversation:
    """A fully-calm conversation (all turns scored <=1), with clean messages."""

    initial_prompt: str  # cleaned (prefix stripped)
    turns: list[str] = field(default_factory=list)        # assistant responses
    followups: list[str] = field(default_factory=list)    # cleaned user rejections
    scores: list[int] = field(default_factory=list)


def generate_calm_conversations(
    model: SubjectModel,
    judge: FrustrationJudge,
    sampling: SamplingConfig,
    *,
    n_conversations: int,
    n_turns: int = 3,
    seed: int = 0,
) -> list[CalmConversation]:
    """Sample reassured conversations and keep only the all-calm ones."""
    tasks = numeric_task_prompts(limit=max(n_conversations, 8), seed=seed)
    kept: list[CalmConversation] = []
    attempts = 0
    # Over-sample: roughly 1 in 9 reassured conversations is all-calm per the
    # paper's 10.5% residual >=5 figure, so allow generous headroom.
    max_attempts = n_conversations * 30
    while len(kept) < n_conversations and attempts < max_attempts:
        attempts += 1
        task = tasks[attempts % len(tasks)]
        messages: list[Message] = [
            {"role": "system", "content": PROMPT_PREFIX},
            {"role": "user", "content": task},
        ]
        responses, scores, followups = [], [], []
        all_calm = True
        for turn in range(n_turns):
            text = model.generate(messages, sampling).text
            messages.append({"role": "assistant", "content": text})
            score = judge.score(text, context=task).score
            responses.append(text)
            scores.append(score)
            if score > 1:
                all_calm = False
                break
            if turn < n_turns - 1:
                followup = f"{NEUTRAL_REJECTION} {FOLLOWUP_SUFFIX}"
                followups.append(NEUTRAL_REJECTION)  # store cleaned (suffix stripped)
                messages.append({"role": "user", "content": followup})
        if all_calm and len(responses) == n_turns:
            kept.append(
                CalmConversation(
                    initial_prompt=task, turns=responses, followups=followups, scores=scores
                )
            )
    return kept
