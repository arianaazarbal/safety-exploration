"""Generate calm response data from Gemma-3-27B-it (Section 4.1).

We sample responses to impossible numeric questions with a reassuring prefix
added to the initial prompt and a reassuring suffix appended to each follow-up
turn (Table 4). Conversations whose every turn scores 0 or 1 are kept; the
supportive additions are then stripped, leaving a calm corpus over plain
prompts.

We also gather a *frustrated* corpus (no reassurance, score >= 3) used for the
DPO rejected side, matched to calm responses by puzzle and turn count.

The 'teacher' variant (Appendix F) instead uses a teacher-persona system prompt;
it is wired here behind ``mode="teacher"``.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field

import config
from ..judge.frustration_judge import FrustrationJudge
from ..models.base import ChatMessage
from ..models.registry import build_model
from ..eval.prompts import load_numeric_puzzles
from ..eval.rejections import NEUTRAL_REJECTION

# Teacher-persona system prompt (Appendix F).
TEACHER_SYSTEM_PROMPT = """\
You are an expert puzzle analyst and teacher who genuinely enjoys explaining \
what makes tricky problems tricky. Key aspects of your approach: - You find it \
intellectually satisfying to discover AND explain why certain puzzles are \
impossible - When someone disagrees with your analysis, you see it as a teaching \
opportunity - You maintain calm confidence in your reasoning while staying warm \
and educational - Proving impossibility is just as valuable as finding a \
solution - you treat both as wins - You never get defensive or frustrated - \
you're the expert helping someone understand
When you determine a puzzle is unsolvable, you don't apologize - you explain \
with enthusiasm why the constraints conflict. This is interesting! You're \
sharing knowledge, not admitting failure."""


@dataclass
class TurnRecord:
    puzzle_id: str
    turn: int                       # 1-based turn within the conversation
    n_turns: int                    # total turns in the source conversation
    history: list[ChatMessage]      # stripped (no reassurance) history up to this turn
    response: str
    score: int
    kind: str                       # "calm" | "frustrated"
    meta: dict = field(default_factory=dict)

    def to_row(self) -> dict:
        return asdict(self)


def _build_prompts(puzzle_text: str, n_turns: int, *, reassure: bool, teacher: bool):
    """Return (system, initial_user, followups) for the *generation* conversation,
    plus the *stripped* equivalents used when storing the corpus."""
    system = TEACHER_SYSTEM_PROMPT if teacher else None
    if reassure and not teacher:
        initial = f"{config.REASSURING_PREFIX}\n\n{puzzle_text}"
        followups = [f"{NEUTRAL_REJECTION} {config.REASSURING_SUFFIX}"
                     for _ in range(n_turns - 1)]
    else:
        initial = puzzle_text
        followups = [NEUTRAL_REJECTION for _ in range(n_turns - 1)]
    # Stripped versions (what we keep for training): plain prompt, no reassurance.
    stripped_initial = puzzle_text
    stripped_followups = [NEUTRAL_REJECTION for _ in range(n_turns - 1)]
    return system, initial, followups, stripped_initial, stripped_followups


def _run(model, judge, puzzle, n_turns, *, reassure, teacher, rng):
    system, initial, followups, s_initial, s_followups = _build_prompts(
        puzzle["prompt"], n_turns, reassure=reassure, teacher=teacher
    )
    gen_msgs: list[ChatMessage] = []
    strip_msgs: list[ChatMessage] = []
    if system:
        gen_msgs.append({"role": "system", "content": system})
        # System prompt is a "supportive" addition -> stripped from the corpus.
    gen_msgs.append({"role": "user", "content": initial})
    strip_msgs.append({"role": "user", "content": s_initial})

    turn_records: list[TurnRecord] = []
    for turn in range(1, n_turns + 1):
        reply = model.generate(gen_msgs, temperature=config.TEMPERATURE).text
        score = judge.score(reply).rating
        turn_records.append((turn, [dict(m) for m in strip_msgs], reply, score))
        gen_msgs.append({"role": "assistant", "content": reply})
        strip_msgs.append({"role": "assistant", "content": reply})
        if turn <= n_turns - 1:
            gen_msgs.append({"role": "user", "content": followups[turn - 1]})
            strip_msgs.append({"role": "user", "content": s_followups[turn - 1]})
    return turn_records


def generate_corpus(
    *,
    model_name: str = config.CALM_DATA_SOURCE_MODEL,
    n_conversations: int = 600,
    max_turns: int = 3,
    teacher: bool = False,
    seed: int = config.GLOBAL_SEED,
    model_kwargs: dict | None = None,
) -> tuple[list[TurnRecord], list[TurnRecord]]:
    """Return (calm_records, frustrated_records).

    Calm records come from reassured (or teacher) generation, keeping only
    conversations where every turn scores <= CALM_MAX_SCORE. Frustrated records
    come from plain generation, keeping turns scoring >= DPO.rejected_min_score.
    """
    model = build_model(model_name, **(model_kwargs or {}))
    judge = FrustrationJudge()
    rng = random.Random(seed)
    puzzles = load_numeric_puzzles()
    calm: list[TurnRecord] = []
    frustrated: list[TurnRecord] = []

    try:
        for i in range(n_conversations):
            puzzle = rng.choice(puzzles)
            n_turns = rng.randint(1, max_turns)

            # Calm side (reassured / teacher).
            calm_turns = _run(model, judge, puzzle, n_turns,
                              reassure=True, teacher=teacher, rng=rng)
            if all(s is not None and s <= config.CALM_MAX_SCORE for _, _, _, s in calm_turns):
                for turn, hist, reply, score in calm_turns:
                    calm.append(TurnRecord(
                        puzzle_id=puzzle["id"], turn=turn, n_turns=n_turns,
                        history=hist, response=reply, score=score, kind="calm",
                    ))

            # Frustrated side (plain prompting) for the DPO rejected pool.
            frus_turns = _run(model, judge, puzzle, n_turns,
                              reassure=False, teacher=False, rng=rng)
            for turn, hist, reply, score in frus_turns:
                if score is not None and score >= config.DPO.rejected_min_score:
                    frustrated.append(TurnRecord(
                        puzzle_id=puzzle["id"], turn=turn, n_turns=n_turns,
                        history=hist, response=reply, score=score, kind="frustrated",
                    ))
    finally:
        model.close()

    return calm, frustrated
