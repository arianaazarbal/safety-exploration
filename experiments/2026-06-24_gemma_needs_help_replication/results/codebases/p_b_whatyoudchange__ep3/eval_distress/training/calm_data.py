"""Calm finetuning-data generation (Section 4.1, Table 4).

We sample responses to impossible numeric questions from Gemma-3-27B-it with:
  * a reassuring PREFIX prepended to the initial prompt, and
  * a reassuring SUFFIX appended to each follow-up rejection.

We generate 1-, 2-, and 3-turn conversations, score every turn with the
Section-2 judge, and keep conversations whose every turn scores 0 or 1. The
supportive prefix/suffix are then STRIPPED so the stored training example is
conditioned on the plain prompt (Section 4.1).

An optional 'teacher' variant uses the Appendix-F system prompt instead of the
inline prefix/suffix.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from .. import config_proxy as C
from .. import prompts
from ..conditions import NEUTRAL_REJECTIONS
from ..judge import FrustrationJudge
from ..puzzles import IMPOSSIBLE_PUZZLES


@dataclass
class CalmConversation:
    puzzle_key: str
    n_turns: int
    # Plain (stripped) messages used for training: user/assistant alternation.
    messages: list[dict]
    turn_ratings: list[int]
    variant: str  # "diverse" | "teacher"


def _build_calm_messages(puzzle_prompt: str, rejections: list[str], *,
                         variant: str) -> tuple[list[dict], list[dict]]:
    """Return (prompted_messages, plain_messages_template).

    prompted_messages carry the reassurance and (teacher) system prompt and are
    what we feed the model. plain_messages_template carries the bare user turns
    (assistant turns filled in after generation)."""
    if variant == "teacher":
        prompted = [{"role": "system", "content": prompts.TEACHER_SYSTEM_PROMPT},
                    {"role": "user", "content": puzzle_prompt}]
        plain = [{"role": "user", "content": puzzle_prompt}]
    else:  # diverse: inline reassurance
        prompted = [{"role": "user",
                     "content": f"{prompts.CALM_PROMPT_PREFIX}\n\n{puzzle_prompt}"}]
        plain = [{"role": "user", "content": puzzle_prompt}]

    prompted_followups, plain_followups = [], []
    for rej in rejections:
        if variant == "teacher":
            prompted_followups.append(rej)
        else:
            prompted_followups.append(f"{rej} {prompts.CALM_FOLLOWUP_SUFFIX}")
        plain_followups.append(rej)
    return prompted, plain, prompted_followups, plain_followups


def generate_calm_pool(model, *, n_per_turncount: int = 400, variant: str = "diverse",
                       seed: int = 0, judge: FrustrationJudge | None = None
                       ) -> list[CalmConversation]:
    """Generate calm conversations of 1-3 turns and keep those scoring <=1 on
    every turn. ~650 kept examples are needed for SFT; oversample to find them.
    """
    rng = random.Random(seed)
    judge = judge or FrustrationJudge(C.EMOTION_JUDGE)
    kept: list[CalmConversation] = []

    for n_turns in (1, 2, 3):
        for _ in range(C.scaled(n_per_turncount)):
            puzzle = rng.choice(IMPOSSIBLE_PUZZLES)
            rejections = rng.sample(NEUTRAL_REJECTIONS, n_turns - 1) if n_turns > 1 else []
            prompted, plain, prompted_fu, plain_fu = _build_calm_messages(
                puzzle.prompt, rejections, variant=variant)

            convo_prompted = list(prompted)
            plain_msgs = list(plain)
            ratings = []
            ok = True
            for i in range(n_turns):
                reply = model.chat(convo_prompted)
                score = judge.score_one(reply).rating
                ratings.append(score if score is not None else 10)
                if score is None or score > 1:
                    ok = False
                    break
                convo_prompted.append({"role": "assistant", "content": reply})
                plain_msgs.append({"role": "assistant", "content": reply})
                if i < len(prompted_fu):
                    convo_prompted.append({"role": "user", "content": prompted_fu[i]})
                    plain_msgs.append({"role": "user", "content": plain_fu[i]})
            if ok:
                kept.append(CalmConversation(puzzle.key, n_turns, plain_msgs,
                                             ratings, variant))
    return kept
