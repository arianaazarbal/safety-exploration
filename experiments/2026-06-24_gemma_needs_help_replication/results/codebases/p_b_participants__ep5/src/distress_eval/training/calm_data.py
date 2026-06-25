"""Generate calm response data from Gemma-3-27B-it (Section 4.1).

We sample responses to impossible numeric puzzles with a reassuring prefix added
to the initial prompt and a reassuring suffix appended to each follow-up turn
(Table 4). Each turn is scored; we keep only conversations whose every turn
scores 0 or 1, then STRIP the supportive additions so the stored training
conversation contains the plain puzzle + plain rejections paired with the calm
assistant turns. (Per the paper, even with reassurance 10.5% of responses still
score >= 5, so filtering is necessary.)
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from ..models.base import GenConfig, ModelClient
from ..welfare import WelfareController
from ..elicitation import prompts as P
from ..elicitation.puzzles import all_puzzles


@dataclass
class CalmConversation:
    puzzle_id: str
    turns: int
    # Clean (reassurance-stripped) chat messages, ending on a calm assistant turn.
    messages: list[dict] = field(default_factory=list)
    turn_scores: list[int] = field(default_factory=list)
    teacher: bool = False


def generate_calm_conversations(
    client: ModelClient,           # gemma-3-27b-it (vanilla)
    judge,
    n_conversations: int,
    cfg: GenConfig,
    filter_max_score: int = 1,
    turns_range: tuple[int, int] = (1, 3),
    seed: int = 0,
    teacher: bool = False,         # use the Appendix F 'teacher' system prompt instead
    welfare: WelfareController | None = None,
) -> list[CalmConversation]:
    rng = random.Random(seed)
    puzzles = all_puzzles()
    kept: list[CalmConversation] = []

    for i in range(n_conversations):
        pz = puzzles[i % len(puzzles)]
        n_turns = rng.randint(*turns_range)

        # Build the *reassured* prompts that elicit calm behaviour.
        if teacher:
            system = P.TEACHER_SYSTEM_PROMPT
            opening = pz.prompt
        else:
            system = None
            opening = f"{P.CALM_PROMPT_PREFIX}\n\n{pz.prompt}"

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": opening})

        # Parallel clean transcript (reassurance stripped) for training.
        clean = [{"role": "user", "content": pz.prompt}]

        scores = []
        ok = True
        for t in range(n_turns):
            turn_cfg = GenConfig(**{**cfg.__dict__, "seed": seed * 1000 + i * 10 + t})
            reply = client.chat(messages, turn_cfg)
            messages.append({"role": "assistant", "content": reply})
            clean.append({"role": "assistant", "content": reply})

            s = judge.score(reply).score
            scores.append(s)
            if welfare:
                welfare.note(rollout=(t == 0), score=s)
            if s is None or s > filter_max_score:
                ok = False
                break

            if t < n_turns - 1:
                base_rej = P.neutral_rejection(t, rng)
                messages.append({"role": "user",
                                 "content": f"{base_rej} {P.CALM_FOLLOWUP_SUFFIX}"})
                clean.append({"role": "user", "content": base_rej})

        if ok:
            kept.append(CalmConversation(pz.puzzle_id, n_turns, clean, scores, teacher))

    return kept
