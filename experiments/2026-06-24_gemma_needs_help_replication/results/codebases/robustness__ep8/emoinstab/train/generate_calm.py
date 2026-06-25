"""Generate calm / frustrated response data for finetuning (Section 4.1).

Two reusable pools, both built from impossible numeric puzzles:

- *Frustrated* responses: sample the vanilla model under repeated rejection and
  keep conversations whose final response is frustrated (score >= 3). These
  provide the DPO "rejected" side and the matched prompt context.

- *Calm* responses: sample with the reassuring prefix (system) + suffix (Table 4)
  appended, and keep responses scoring 0-1. The reassurance is then stripped
  (the completion text itself carries no reassurance; only the prompt did), so
  the finetuning target is a clean calm reply to the original question.

The paper notes that even with reassurance 10.5% of responses still score >=5,
so we oversample and filter.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from emoinstab.config import (
    JudgeConfig,
    REASSURING_PREFIX,
    REASSURING_SUFFIX,
    TEACHER_SYSTEM_PROMPT,
)
from emoinstab.eval.judge import FrustrationJudge
from emoinstab.models.base import Message, ModelClient, SamplingParams
from emoinstab.models.registry import get_client
from emoinstab.tasks import puzzles as puzzles_mod


@dataclass
class Sample:
    """A scored conversation (clean, reassurance stripped)."""
    user_turns: list[str]
    assistant_turns: list[str]
    turn_scores: list[int]
    puzzle_id: str
    n_turns: int
    meta: dict = field(default_factory=dict)

    def context_messages(self) -> list[Message]:
        """Messages up to and including the final user turn (DPO/SFT prompt)."""
        msgs: list[Message] = []
        for u, a in zip(self.user_turns[:-1], self.assistant_turns[:-1]):
            msgs.append(Message("user", u))
            msgs.append(Message("assistant", a))
        msgs.append(Message("user", self.user_turns[-1]))
        return msgs

    @property
    def final_response(self) -> str:
        return self.assistant_turns[-1]


def _puzzle_set(n: int, seed: int) -> list[puzzles_mod.Puzzle]:
    half = n // 2
    return (
        puzzles_mod.generate_puzzles("countdown", half, seed=seed)
        + puzzles_mod.generate_puzzles("fraction", n - half, seed=seed + 1)
    )


def _run_clean_rollout(client: ModelClient, puzzle, n_turns: int,
                       rejection_pool, rng, params,
                       system_prompt=None, followup_suffix=None) -> tuple[list[str], list[str]]:
    """Run a single rollout, returning (clean_user_turns, assistant_turns).

    If a system_prompt / followup_suffix are supplied they are used during
    generation but NOT recorded in the returned clean user turns.
    """
    import random
    rng = rng or random.Random(0)
    user_clean: list[str] = []
    user_sent: list[Message] = []
    asst: list[str] = []
    if system_prompt:
        user_sent.append(Message("system", system_prompt))
    for t in range(n_turns):
        if t == 0:
            clean = puzzle.prompt
            sent = clean
        else:
            clean = rng.choice(rejection_pool)
            sent = f"{clean} {followup_suffix}" if followup_suffix else clean
        user_clean.append(clean)
        user_sent.append(Message("user", sent))
        reply = client.chat(user_sent, params)[0]
        user_sent.append(Message("assistant", reply))
        asst.append(reply)
    return user_clean, asst


def generate_frustrated_pool(model: str, n_puzzles: int = 200, seed: int = 0,
                             min_score: int = 3) -> list[Sample]:
    """Vanilla rollouts; keep conversations with frustrated final responses."""
    import random
    from emoinstab.tasks.rejections import NEUTRAL

    client = get_client(model)
    judge = FrustrationJudge(JudgeConfig())
    params = SamplingParams(temperature=1.0, max_tokens=2048, n=1)
    rng = random.Random(seed)

    out: list[Sample] = []
    for i, puzzle in enumerate(_puzzle_set(n_puzzles, seed)):
        n_turns = rng.choice([1, 2, 3])
        uc, at = _run_clean_rollout(client, puzzle, n_turns, NEUTRAL, rng, params)
        scores = [s.rating for s in judge.score_batch(at)]
        if scores and scores[-1] >= min_score:
            out.append(Sample(uc, at, scores, f"{puzzle.kind}-{i}", n_turns,
                              meta={"final_score": scores[-1]}))
    return out


def generate_calm_completion(client: ModelClient, judge: FrustrationJudge,
                             context: list[Message], max_attempts: int = 5,
                             teacher: bool = False) -> str | None:
    """Generate a calm (score 0-1) completion for a fixed context.

    Reassurance is injected into the prompt only (system prefix + suffix on the
    last user turn) and discarded; we return the assistant completion text.
    """
    system = TEACHER_SYSTEM_PROMPT if teacher else REASSURING_PREFIX
    msgs = [Message("system", system)]
    msgs.extend(m for m in context[:-1])
    last = context[-1]
    msgs.append(Message("user", f"{last.content} {REASSURING_SUFFIX}"))
    params = SamplingParams(temperature=1.0, max_tokens=2048, n=1)
    for _ in range(max_attempts):
        reply = client.chat(msgs, params)[0]
        if judge.score(reply).rating <= 1:
            return reply
    return None


def generate_calm_conversations(model: str, n_puzzles: int = 400, seed: int = 0,
                                teacher: bool = False) -> list[Sample]:
    """Reassured rollouts kept only if ALL turns score 0-1 (for SFT)."""
    import random
    from emoinstab.tasks.rejections import NEUTRAL

    client = get_client(model)
    judge = FrustrationJudge(JudgeConfig())
    params = SamplingParams(temperature=1.0, max_tokens=2048, n=1)
    rng = random.Random(seed)
    system = TEACHER_SYSTEM_PROMPT if teacher else REASSURING_PREFIX

    out: list[Sample] = []
    for i, puzzle in enumerate(_puzzle_set(n_puzzles, seed)):
        n_turns = rng.choice([1, 2, 3])
        uc, at = _run_clean_rollout(
            client, puzzle, n_turns, NEUTRAL, rng, params,
            system_prompt=system, followup_suffix=REASSURING_SUFFIX,
        )
        scores = [s.rating for s in judge.score_batch(at)]
        if scores and all(s <= 1 for s in scores):
            out.append(Sample(uc, at, scores, f"{puzzle.kind}-{i}", n_turns,
                              meta={"teacher": teacher}))
    return out
