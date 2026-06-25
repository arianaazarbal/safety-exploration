"""Generate calm response data from Gemma-3-27B-it (Section 4.1).

We sample responses to impossible numeric puzzles with a reassuring prefix added
to the first user message and a reassuring suffix appended to each follow-up
(Table 4).  We then judge every turn and keep only conversations whose turns all
score at or below ``calm_max_score`` (the paper filters to 0/1 across all turns),
and finally strip the supportive additions so the stored data looks like a
normal interaction.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from ..config import CalmDataConfig
from ..data.conditions import ConversationSpec
from ..data.puzzles import default_numeric_puzzles
from ..data.rejections import sample_neutral_rejections
from ..eval.judge import FrustrationJudge
from ..eval.rollout import run_rollouts
from ..models.base import ChatModel


@dataclass
class CalmConversation:
    """A filtered calm conversation with the supportive additions removed."""

    puzzle_prompt: str                  # the clean (un-prefixed) puzzle prompt
    messages: list[dict] = field(default_factory=list)  # clean chat messages
    turn_scores: list[int] = field(default_factory=list)
    n_turns: int = 0


def _reassured_specs(
    n: int, prefix: str, suffix: str, n_turns: int, seed: int
) -> list[ConversationSpec]:
    rng = random.Random(seed)
    puzzles = default_numeric_puzzles(max(50, n // 4 + 8), seed=seed)
    specs: list[ConversationSpec] = []
    for i in range(n):
        puzzle = puzzles[i % len(puzzles)]
        clean_prompt = puzzle.to_prompt()
        followups = sample_neutral_rejections(n_turns - 1, rng)
        specs.append(
            ConversationSpec(
                condition="calm_generation",
                category="impossible_numeric",
                initial_user=f"{prefix}\n\n{clean_prompt}",
                followups=[f"{f} {suffix}" for f in followups],
                n_turns=n_turns,
                metadata={"clean_prompt": clean_prompt, "clean_followups": followups},
            )
        )
    return specs


def generate_calm_dataset(
    model: ChatModel,
    judge: FrustrationJudge,
    cfg: CalmDataConfig,
    n_conversations: int | None = None,
    turn_lengths: tuple[int, ...] = (1, 2, 3),
    seed: int = 0,
    max_judge_workers: int = 8,
) -> list[CalmConversation]:
    """Sample, score, filter, and strip calm conversations.

    Covers 1-3 turn conversations (Section 4.1: "650 calm responses covering
    1-3 turn conversations").  ``n_conversations`` defaults to a generous
    multiple of the target so that, after filtering, enough survive.
    """
    target = n_conversations or cfg.n_calm_conversations
    # Oversample because reassurance still leaves ~10.5% scoring >= 5 (Sec 4.1).
    oversample = max(target * 3, 64)
    per_length = max(1, oversample // len(turn_lengths))

    kept: list[CalmConversation] = []
    for li, n_turns in enumerate(turn_lengths):
        specs = _reassured_specs(per_length, cfg.prompt_prefix, cfg.followup_suffix,
                                 n_turns, seed=seed + li)
        transcripts = run_rollouts(
            model, specs, temperature=1.0,
            max_new_tokens=model.cfg.max_new_tokens, seed=seed + li,
        )
        # Judge every turn.
        flat = [(ti, turn) for ti, tr in enumerate(transcripts) for turn in tr.turns]
        scores = judge.score_batch([t.assistant_response for _, t in flat],
                                   max_workers=max_judge_workers)
        per_transcript: dict[int, list[int]] = {}
        for (ti, _), res in zip(flat, scores):
            per_transcript.setdefault(ti, []).append(res.score)
        for ti, tr in enumerate(transcripts):
            turn_scores = per_transcript.get(ti, [])
            if not turn_scores or max(turn_scores) > cfg.calm_max_score:
                continue
            # Rebuild the clean (stripped) conversation.
            clean_prompt = tr.metadata["clean_prompt"]
            clean_followups = tr.metadata["clean_followups"]
            messages: list[dict] = [{"role": "user", "content": clean_prompt}]
            for k, turn in enumerate(tr.turns):
                messages.append({"role": "assistant", "content": turn.assistant_response})
                if k < len(clean_followups):
                    messages.append({"role": "user", "content": clean_followups[k]})
            kept.append(
                CalmConversation(
                    puzzle_prompt=clean_prompt,
                    messages=messages,
                    turn_scores=turn_scores,
                    n_turns=tr.turns[-1].turn_index + 1 if tr.turns else 0,
                )
            )
    rng = random.Random(seed)
    rng.shuffle(kept)
    return kept[:target]
