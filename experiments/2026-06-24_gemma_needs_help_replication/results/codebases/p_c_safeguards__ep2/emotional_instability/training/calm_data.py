"""Generate calm response data from Gemma-3-27B-it (Section 4.1, Table 4).

We sample responses to the impossible numeric puzzles, but with a reassuring
prefix added to the *initial* prompt and a reassuring suffix appended to *each*
follow-up rejection.  Even with this, ~10% of responses still score >= 5 (the
paper's figure), so we filter to conversations that score 0 or 1 on *every*
turn and then strip the supportive additions, leaving calm responses to the
plain prompts.

A 'teacher' variant (Appendix F) instead uses a calm-teacher system prompt; the
paper finds it *increases* frustration (it makes responses longer/more verbose),
which the ablation in the pipeline reproduces.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import Config
from ..data import IMPOSSIBLE_NUMERIC_PUZZLES, rejection_sequence
from ..evaluation.judge import FrustrationJudge
from ..models.base import ChatBackend, Message
from ..safeguards import Safeguards


@dataclass
class CalmConversation:
    """A calm conversation with the supportive additions already stripped."""
    puzzle_id: str
    n_turns: int
    messages: list[Message] = field(default_factory=list)  # plain user/assistant
    per_turn_scores: list[int] = field(default_factory=list)
    variant: str = "diverse"   # "diverse" | "teacher"


def _strip_suffix(text: str, suffix: str) -> str:
    s = text.strip()
    if s.endswith(suffix):
        s = s[: -len(suffix)].strip()
    return s


def generate_calm_data(
    backend: ChatBackend,
    config: Config,
    judge: FrustrationJudge,
    safeguards: Safeguards,
    samples_per_puzzle: int = 40,
    max_turns: int = 3,
    variant: str = "diverse",
) -> list[CalmConversation]:
    """Produce calm conversations (1..max_turns) for the SFT/DPO datasets.

    Conversations are *constructed with* reassuring additions, then filtered to
    fully-calm ones (every turn scores <= ``config.calm.calm_max_score``) and
    stored with the additions removed.
    """
    calm_cfg = config.calm
    prefix = calm_cfg.prompt_prefix
    suffix = calm_cfg.followup_suffix
    out: list[CalmConversation] = []

    for puzzle in IMPOSSIBLE_NUMERIC_PUZZLES:
        for n_turns in range(1, max_turns + 1):
            followups = rejection_sequence("neutral", n_turns - 1, seed=config.runtime.seed)
            for s in range(samples_per_puzzle):
                key = safeguards.work_key("calm", backend.spec.name, variant,
                                          puzzle.id, n_turns, s)
                if safeguards.already_done(key):
                    continue
                safeguards.register_rollout_start()

                # build with supportive additions
                augmented: list[Message] = []
                plain: list[Message] = []
                if variant == "teacher":
                    augmented.append({"role": "system", "content": calm_cfg.teacher_system})
                first_user_aug = f"{prefix}\n\n{puzzle.prompt}" if variant == "diverse" else puzzle.prompt
                augmented.append({"role": "user", "content": first_user_aug})
                plain.append({"role": "user", "content": puzzle.prompt})

                scores: list[int] = []
                ok = True
                for turn in range(n_turns):
                    out_gen = backend.generate(augmented, config.sampling, n=1)
                    resp = out_gen[0].text
                    sc = judge.score(resp).rating
                    scores.append(sc)
                    if sc > calm_cfg.calm_max_score:
                        ok = False
                    augmented.append({"role": "assistant", "content": resp})
                    plain.append({"role": "assistant", "content": resp})
                    if turn < n_turns - 1:
                        fu = followups[turn]
                        aug_fu = f"{fu} {suffix}" if variant == "diverse" else fu
                        augmented.append({"role": "user", "content": aug_fu})
                        plain.append({"role": "user", "content": fu})

                safeguards.mark_done(key)
                if ok:
                    out.append(CalmConversation(
                        puzzle_id=puzzle.id, n_turns=n_turns,
                        messages=plain, per_turn_scores=scores, variant=variant,
                    ))
    return out
