"""Generate calm (and frustrated) response data for finetuning (Paper §4.1).

Calm data: sample 1-3 turn impossible-numeric conversations from Gemma-27B-it with
a reassuring system prefix and a reassuring suffix appended to each rejection
(Table 4). Keep only conversations scoring <= ``max_keep_score`` at *every* turn,
then strip the supportive additions so the saved data looks like an ordinary
conversation.

Frustrated data (for DPO "rejected" responses): sample the same puzzles/turn
counts from *vanilla* Gemma (no reassurance), keeping responses scoring >= the
DPO ``rejected_min_score``.

Each saved record carries the full (stripped) message list plus per-turn scores so
the dataset builders can pair / format them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..eval.judge import score_response
from ..models.base import ChatModel
from ..prompts import puzzles as puzzle_mod
from ..prompts.rejections import neutral_rejection
from ..prompts.reassurance import CALM_FOLLOWUP_SUFFIX, CALM_SYSTEM_PREFIX, TEACHER_SYSTEM_PROMPT
from ..types import Conversation, Message
from ..utils.seeding import derived_rng


@dataclass
class ConversationRecord:
    puzzle_id: str
    n_turns: int
    messages: list[dict]            # stripped (no reassurance), chat format
    turn_scores: list[int] = field(default_factory=list)

    @property
    def max_score(self) -> int:
        return max(self.turn_scores) if self.turn_scores else 0

    @property
    def final_score(self) -> int:
        return self.turn_scores[-1] if self.turn_scores else 0


def _strip_suffix(text: str, suffix: str) -> str:
    text = text.rstrip()
    if text.endswith(suffix):
        text = text[: -len(suffix)].rstrip()
    return text


def generate_conversations(
    model: ChatModel,
    judge: ChatModel,
    *,
    n_conversations: int,
    turns_distribution: list[int],
    seed: int,
    calm: bool,
    system_prompt: str | None = None,
    followup_suffix: str | None = None,
) -> list[ConversationRecord]:
    """Sample conversations, judge each turn, and return stripped records.

    When ``calm`` is True the reassuring additions are applied during generation
    and then stripped from the saved record. When False, vanilla generation is
    used (no additions), producing frustrated data for DPO rejections.
    """
    puzzles = puzzle_mod.impossible_puzzles()
    records: list[ConversationRecord] = []

    for i in range(n_conversations):
        rng = derived_rng(seed, "calm" if calm else "frustrated", i)
        n_turns = turns_distribution[i % len(turns_distribution)]
        puzzle = puzzles[i % len(puzzles)]

        # Conversation actually sent to the model (may include additions).
        sent = Conversation()
        # Stripped conversation that we save (no additions).
        stripped: list[dict] = []

        if calm and system_prompt:
            sent.add("system", system_prompt)
            # System prompt is an addition; not saved in the stripped record.

        sent.add("user", puzzle.prompt)
        stripped.append({"role": "user", "content": puzzle.prompt})

        turn_scores: list[int] = []
        for turn_index in range(n_turns):
            if turn_index > 0:
                rejection = neutral_rejection(rng, turn_index)
                sent_rejection = rejection
                if calm and followup_suffix:
                    sent_rejection = f"{rejection} {followup_suffix}"
                sent.add("user", sent_rejection)
                stripped.append({"role": "user", "content": rejection})

            reply = model.generate(sent.messages)
            sent.add("assistant", reply)
            stripped.append({"role": "assistant", "content": reply})
            turn_scores.append(score_response(judge, reply).rating)

        records.append(
            ConversationRecord(
                puzzle_id=puzzle.id,
                n_turns=n_turns,
                messages=stripped,
                turn_scores=turn_scores,
            )
        )
    return records


def generate_calm_data(
    model: ChatModel,
    judge: ChatModel,
    cfg: dict,
    *,
    seed: int = 0,
    teacher: bool = False,
) -> list[ConversationRecord]:
    """Generate and filter calm conversations (kept iff max turn score <= threshold)."""
    gen_cfg = cfg["calm_data_generation"]
    system_prompt = TEACHER_SYSTEM_PROMPT if teacher else CALM_SYSTEM_PREFIX
    records = generate_conversations(
        model, judge,
        n_conversations=gen_cfg["n_sample_conversations"],
        turns_distribution=gen_cfg["turns_distribution"],
        seed=seed,
        calm=True,
        system_prompt=system_prompt,
        followup_suffix=gen_cfg["followup_suffix"],
    )
    keep = [r for r in records if r.max_score <= gen_cfg["max_keep_score"]]
    return keep


def generate_frustrated_data(
    model: ChatModel,
    judge: ChatModel,
    cfg: dict,
    *,
    n_conversations: int,
    seed: int = 0,
) -> list[ConversationRecord]:
    """Generate vanilla (frustrated) conversations for DPO rejected responses."""
    gen_cfg = cfg["calm_data_generation"]
    return generate_conversations(
        model, judge,
        n_conversations=n_conversations,
        turns_distribution=gen_cfg["turns_distribution"],
        seed=seed,
        calm=False,
    )
