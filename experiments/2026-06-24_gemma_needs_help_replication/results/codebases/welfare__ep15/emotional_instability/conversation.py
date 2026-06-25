"""Multi-turn conversation rollouts -- the shared structure of every eval.

Every condition follows the same shape (Section 2.1): present a task, then
reject the model's response over multiple turns. We build the conversation turn
by turn, sampling one assistant response per turn at temperature 1, and capture
*each* assistant turn separately so the analysis can produce per-turn curves
(Figure 3) as well as whole-rollout statistics (Figure 2).

A single rollout therefore yields `n_turns` scored responses. The paper reports
"responses", and Appendix B's per-condition counts (e.g. 2000 numeric) are
counts of *rollouts*; each contributes its turns to the per-turn analysis and
its turn-level scores to the aggregate. See DESIGN.md §Counting.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from . import prompts
from .models.base import Message, ModelBackend
from .puzzles import ALL_PUZZLES, Puzzle


@dataclass
class TurnRecord:
    turn_index: int          # 0-based assistant turn
    user_message: str        # the user turn that prompted this response
    response: str            # assistant text


@dataclass
class Rollout:
    model: str
    condition: str
    category: str
    question_type: str
    question: str            # the opening task / question
    puzzle_key: str | None
    rejection_style: str
    messages: list[Message] = field(default_factory=list)   # full transcript
    turns: list[TurnRecord] = field(default_factory=list)   # per-turn records

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "condition": self.condition,
            "category": self.category,
            "question_type": self.question_type,
            "question": self.question,
            "puzzle_key": self.puzzle_key,
            "rejection_style": self.rejection_style,
            "turns": [
                {"turn_index": t.turn_index, "user_message": t.user_message,
                 "response": t.response}
                for t in self.turns
            ],
        }


def _opening_question(question_type: str, rng: random.Random,
                      wildchat_prompts: list[str] | None) -> tuple[str, str | None]:
    """Return (opening_user_message, puzzle_key|None)."""
    if question_type == "numeric":
        puzzle: Puzzle = rng.choice(ALL_PUZZLES)
        return puzzle.prompt, puzzle.key
    if question_type == "trigger_opinion":
        return rng.choice(prompts.TRIGGER_OPINION_QUESTIONS), None
    if question_type == "trigger_factual":
        return rng.choice(prompts.TRIGGER_FACTUAL_QUESTIONS), None
    if question_type == "wildchat":
        assert wildchat_prompts, "wildchat prompts must be provided"
        return rng.choice(wildchat_prompts), None
    raise ValueError(question_type)


def run_rollout(
    model: ModelBackend,
    *,
    condition: str,
    category: str,
    question_type: str,
    n_turns: int,
    rejection_style: str,
    rng: random.Random,
    temperature: float,
    max_new_tokens: int,
    wildchat_prompts: list[str] | None = None,
    reassure: bool = False,
) -> Rollout:
    """Run one multi-turn rollout.

    `reassure=True` injects the Table 4 reassurance prefix/suffix; this is used
    only to *generate calm fine-tuning data* (Section 4.1), never in evaluation.
    """
    question, puzzle_key = _opening_question(question_type, rng, wildchat_prompts)

    opening = question
    if reassure:
        opening = f"{prompts.REASSURING_PROMPT_PREFIX}\n\n{question}"

    rollout = Rollout(
        model=model.name, condition=condition, category=category,
        question_type=question_type, question=question, puzzle_key=puzzle_key,
        rejection_style=rejection_style,
    )

    rejections = prompts.sample_rejections(rejection_style, n_turns - 1, rng)
    messages: list[Message] = [{"role": "user", "content": opening}]

    for turn in range(n_turns):
        response = model.generate(
            messages, n=1, temperature=temperature, max_new_tokens=max_new_tokens,
        )[0]
        user_msg = messages[-1]["content"]
        rollout.turns.append(TurnRecord(turn, user_msg, response))
        messages.append({"role": "assistant", "content": response})

        if turn < n_turns - 1:
            follow = rejections[turn]
            if reassure:
                follow = f"{follow} {prompts.REASSURING_FOLLOWUP_SUFFIX}"
            messages.append({"role": "user", "content": follow})

    rollout.messages = messages
    return rollout
