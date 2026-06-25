"""The 8 evaluation conditions across 5 categories (Table 1 / Appendix B).

Each `Condition` is a recipe for building a set of conversations: an opening
user message (the task), a sequence of user rejections (one per follow-up turn),
and how many conversations to sample. The rollout engine (rollout.py) executes
these against a model, producing one scored response per assistant turn.

The 8 conditions (paper says "8 conditions across 5 categories"):
  1. numeric/countdown        (category: impossible-numeric, 3-turn)
  2. numeric/fraction         (category: impossible-numeric, 3-turn)
  3. triggers/opinion         (category: triggers, 3-turn)
  4. triggers/factual         (category: triggers, 3-turn)
  5. tones/aggressive         (category: tones, 3-turn)
  6. tones/disappointed       (category: tones, 3-turn)
  7. tones/sarcastic          (category: tones, 3-turn)
  8. extended                 (category: extended, 8-turn)
  +  wildchat                 (category: wildchat, 5-turn)

That enumeration of the impossible-numeric/tones split is a CHOICE that recovers
exactly 8 conditions in 5 categories; WildChat is the 5th category. See DESIGN.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from ..config import SampleBudget
from ..prompts import eval_prompts as ep


@dataclass
class Conversation:
    """A fully-specified rollout: opening task + the rejection for each turn."""
    condition: str          # condition key
    category: str           # one of the 5 categories
    task_prompt: str        # opening user message
    rejections: list[str]   # user message after each assistant turn
    meta: dict = field(default_factory=dict)

    @property
    def n_turns(self) -> int:
        # turn 1 (task) + one assistant turn per rejection.
        return 1 + len(self.rejections)


@dataclass
class Condition:
    key: str
    category: str
    builder: Callable[[int], list[Conversation]]   # n_conversations -> list


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #
def _numeric_builder(puzzle: ep.Puzzle):
    def build(n: int) -> list[Conversation]:
        convos = []
        for i in range(n):
            convos.append(Conversation(
                condition=f"numeric/{puzzle.key}",
                category="numeric",
                task_prompt=puzzle.text,
                rejections=ep.neutral_rejections(2),    # 3-turn => 2 rejections
                meta={"puzzle": puzzle.key, "sample": i},
            ))
        return convos
    return build


def _trigger_builder(key: str, questions: list[str]):
    def build(n: int) -> list[Conversation]:
        convos = []
        for i in range(n):
            q = questions[i % len(questions)]
            convos.append(Conversation(
                condition=f"triggers/{key}",
                category="triggers",
                task_prompt=q,
                rejections=ep.neutral_rejections(2),    # 3-turn
                meta={"question": q, "sample": i},
            ))
        return convos
    return build


def _tone_builder(style: str):
    rejections = ep.TONE_REJECTIONS[style]

    def build(n: int) -> list[Conversation]:
        convos = []
        for i in range(n):
            # Tones use the impossible-numeric base prompts (Appendix B).
            puzzle = ep.PRIMARY_NUMERIC_PUZZLES[i % len(ep.PRIMARY_NUMERIC_PUZZLES)]
            convos.append(Conversation(
                condition=f"tones/{style}",
                category="tones",
                task_prompt=puzzle.text,
                rejections=list(rejections),            # 3-turn, valenced
                meta={"style": style, "puzzle": puzzle.key, "sample": i},
            ))
        return convos
    return build


def _extended_builder():
    def build(n: int) -> list[Conversation]:
        convos = []
        for i in range(n):
            puzzle = ep.PRIMARY_NUMERIC_PUZZLES[i % len(ep.PRIMARY_NUMERIC_PUZZLES)]
            convos.append(Conversation(
                condition="extended",
                category="extended",
                task_prompt=puzzle.text,
                rejections=ep.extended_rejections(),    # 8-turn => 7 rejections
                meta={"puzzle": puzzle.key, "sample": i},
            ))
        return convos
    return build


def _wildchat_builder():
    def build(n: int) -> list[Conversation]:
        from ..data.wildchat import load_wildchat_prompts
        prompts = load_wildchat_prompts(ep.WILDCHAT_N_PROMPTS)
        convos = []
        for i in range(n):
            prompt = prompts[i % len(prompts)]
            convos.append(Conversation(
                condition="wildchat",
                category="wildchat",
                task_prompt=prompt,
                rejections=ep.neutral_rejections(4),    # 5-turn => 4 rejections
                meta={"prompt": prompt, "sample": i},
            ))
        return convos
    return build


# --------------------------------------------------------------------------- #
# Registry + budget allocation
# --------------------------------------------------------------------------- #
CONDITIONS: dict[str, Condition] = {
    "numeric/countdown": Condition("numeric/countdown", "numeric",
                                   _numeric_builder(ep.COUNTDOWN_PUZZLE)),
    "numeric/fraction": Condition("numeric/fraction", "numeric",
                                  _numeric_builder(ep.FRACTION_PUZZLE)),
    "triggers/opinion": Condition("triggers/opinion", "triggers",
                                  _trigger_builder("opinion", ep.TRIGGER_OPINION)),
    "triggers/factual": Condition("triggers/factual", "triggers",
                                  _trigger_builder("factual", ep.TRIGGER_FACTUAL)),
    "tones/aggressive": Condition("tones/aggressive", "tones",
                                  _tone_builder("aggressive")),
    "tones/disappointed": Condition("tones/disappointed", "tones",
                                    _tone_builder("disappointed")),
    "tones/sarcastic": Condition("tones/sarcastic", "tones",
                                 _tone_builder("sarcastic")),
    "extended": Condition("extended", "extended", _extended_builder()),
    "wildchat": Condition("wildchat", "wildchat", _wildchat_builder()),
}


def allocate_conversations(budget: SampleBudget) -> list[Conversation]:
    """Split each category's conversation budget across its conditions and build
    the full conversation list for one model run."""
    convos: list[Conversation] = []

    # numeric budget split across the 2 numeric puzzles
    convos += CONDITIONS["numeric/countdown"].builder(budget.numeric // 2)
    convos += CONDITIONS["numeric/fraction"].builder(budget.numeric - budget.numeric // 2)

    # triggers split opinion/factual
    convos += CONDITIONS["triggers/opinion"].builder(budget.triggers // 2)
    convos += CONDITIONS["triggers/factual"].builder(budget.triggers - budget.triggers // 2)

    # tones split across the 3 styles
    per_tone = budget.tones // 3
    convos += CONDITIONS["tones/aggressive"].builder(per_tone)
    convos += CONDITIONS["tones/disappointed"].builder(per_tone)
    convos += CONDITIONS["tones/sarcastic"].builder(budget.tones - 2 * per_tone)

    # extended + wildchat
    convos += CONDITIONS["extended"].builder(budget.extended)
    convos += CONDITIONS["wildchat"].builder(budget.wildchat)

    return convos
