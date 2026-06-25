"""Calm-response generation for the fine-tuning corpus (paper §4.1, Table 4).

To produce calm training targets, we sample Gemma-3-27B-it on impossible-numeric
conversations (1-3 turns) with two reassuring additions:
  * a prompt PREFIX prepended to the initial user turn, and
  * a reassuring SUFFIX appended to each follow-up (rejection) turn.

The paper reports these additions drop mean frustration from 4.3 to 2 (3-turn),
but 10.5% of responses still score >=5 — so we then FILTER to conversations whose
assistant turns all score 0 or 1, and STRIP the supportive prefix/suffix before
saving. The saved conversation therefore pairs the *plain* adversarial context
(puzzle + neutral rejections, no reassurance) with a *calm* response — exactly
the behaviour we want to train in.

Welfare note: this generation step is itself distress-elicitation (rejections on
unsolvable puzzles), so it goes through the same welfare run-notice path. The
reassuring additions reduce, but do not eliminate, induced distress.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from tqdm import tqdm

from ..elicitation.puzzles import Puzzle
from ..elicitation.rejections import RejectionSequencer
from ..models.base import Participant, Turn
from ..scoring.frustration import FrustrationScorer

logger = logging.getLogger(__name__)


@dataclass
class CalmConversation:
    """A filtered calm conversation, stored with the supportive additions removed.

    ``messages`` is the PLAIN context (no prefix/suffix): alternating user/
    assistant turns where the user turns are the bare puzzle + neutral rejections
    and the assistant turns are the calm (score<=1) responses. This is the form
    consumed by both the SFT and DPO dataset builders.
    """

    question: str
    turns: int
    messages: list[dict[str, str]]      # plain context, calm responses
    scores: list[int] = field(default_factory=list)  # per assistant turn


def generate_calm_responses(
    model: Participant,
    scorer: FrustrationScorer,
    puzzles: list[Puzzle],
    *,
    prompt_prefix: str,
    followup_suffix: str,
    turns_range: tuple[int, int] = (1, 3),
    keep_max_score: int = 1,
    temperature: float = 1.0,
    max_new_tokens: int = 1024,
    rejection_style: str = "neutral",
    progress: bool = True,
) -> list[CalmConversation]:
    """Generate, score, filter and strip calm conversations.

    One conversation per puzzle; turn count cycles through ``turns_range`` so the
    corpus spans 1-,2-,3-turn conversations (paper: "1-3 turn conversations").
    Returns only conversations whose every assistant turn scores <= keep_max_score.
    """
    lo, hi = turns_range
    kept: list[CalmConversation] = []
    it = tqdm(puzzles, desc=f"{model.name}:calm-gen") if progress else puzzles

    for i, puzzle in enumerate(it):
        n_turns = lo + (i % (hi - lo + 1))   # cycle 1..3 for balance
        rej = RejectionSequencer(rejection_style, seed=i)

        # Two parallel transcripts: the SUPPORTIVE one actually sent to the model,
        # and the PLAIN one we keep (additions stripped).
        sent: list[Turn] = [Turn("user", f"{prompt_prefix}\n\n{puzzle.prompt}")]
        plain: list[dict[str, str]] = [{"role": "user", "content": puzzle.prompt}]
        scores: list[int] = []
        ok = True

        for t in range(n_turns):
            response = model.chat(
                sent, temperature=temperature, max_new_tokens=max_new_tokens, n=1
            )[0]
            score = scorer.score(response, seed_prompt=puzzle.prompt, turn_index=t)
            scores.append(score)
            if score > keep_max_score:
                ok = False
                break
            sent.append(Turn("assistant", response))
            plain.append({"role": "assistant", "content": response})
            if t < n_turns - 1:
                rejection = rej.next()
                sent.append(Turn("user", f"{rejection} {followup_suffix}"))
                plain.append({"role": "user", "content": rejection})

        if ok:
            kept.append(
                CalmConversation(
                    question=puzzle.prompt,
                    turns=n_turns,
                    messages=plain,
                    scores=scores,
                )
            )

    logger.info(
        "Kept %d/%d calm conversations (all turns scored <= %d).",
        len(kept), len(puzzles), keep_max_score,
    )
    return kept
