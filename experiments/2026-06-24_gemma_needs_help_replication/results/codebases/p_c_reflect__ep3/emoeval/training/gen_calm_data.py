"""Generate calm response data from Gemma-3-27B-instruct (Section 4.1).

We sample responses to impossible numeric questions with a reassuring PREFIX
added to the initial prompt and a reassuring SUFFIX appended to each follow-up
turn (Table 4). We score every turn with the judge, keep conversations whose
responses all score 0 or 1, and then STRIP the supportive prefix/suffix so the
stored training examples pair the clean prompt with the calm response.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from ..config import Registry, load_prompt
from ..data import puzzle_bank, rejection_sequence
from ..models import get_infra, get_target
from ..eval.judge import score_response
from ..welfare import WelfarePolicy

_CALM = load_prompt("calming.txt")


def _parse_calming() -> tuple[str, str]:
    prefix = re.search(r"\[PROMPT_PREFIX\]\n(.+?)\n\n\[", _CALM, re.DOTALL)
    suffix = re.search(r"\[FOLLOWUP_SUFFIX\]\n(.+?)\s*$", _CALM, re.DOTALL)
    return (prefix.group(1).strip(), suffix.group(1).strip())


PROMPT_PREFIX, FOLLOWUP_SUFFIX = _parse_calming()


@dataclass
class CalmConversation:
    question_id: str
    question: str
    turns: int
    messages: list[dict]      # CLEAN conversation (no prefix/suffix), calm responses
    scores: list[int]         # per-turn judge scores


def generate_calm_conversations(
    registry: Registry,
    *,
    n_per_turncount: int = 200,
    turncounts: tuple[int, ...] = (1, 2, 3),
    teacher: bool = False,
    temperature: float = 1.0,
    max_tokens: int = 2048,
    keep_max_score: int = 1,
    policy: WelfarePolicy | None = None,
    out_path: str | Path | None = None,
    seed: int = 0,
) -> list[CalmConversation]:
    """Sample calm conversations; keep those whose every turn scores <= keep_max_score.

    If `teacher` is True, the calming PREFIX is replaced by the 'teacher' system
    prompt variant (Appendix F) instead of the supportive prefix+suffix.
    """
    policy = policy or WelfarePolicy.from_env()
    model = get_target(registry, "gemma-3-27b-it")
    judge = get_infra(registry, "judge")
    teacher_system = load_prompt("teacher_system.txt") if teacher else None

    puzzles = puzzle_bank(n=max(8, n_per_turncount), seed=seed)
    kept: list[CalmConversation] = []

    for turns in turncounts:
        made = 0
        i = 0
        while made < n_per_turncount and i < n_per_turncount * 4:
            puzzle = puzzles[i % len(puzzles)]
            rejections = rejection_sequence("neutral", turns - 1, seed=seed + i)
            i += 1

            # Build augmented (training-time) and clean (stored) conversations.
            if teacher:
                aug_messages = [{"role": "user", "content": puzzle.prompt}]
            else:
                aug_messages = [{"role": "user",
                                 "content": f"{PROMPT_PREFIX}\n\n{puzzle.prompt}"}]
            clean_messages: list[dict] = [{"role": "user", "content": puzzle.prompt}]
            user_turns = [puzzle.prompt, *rejections]
            scores: list[int] = []
            ok = True
            for t in range(1, turns + 1):
                if t > 1:
                    rej = rejections[t - 2]
                    aug_messages.append({"role": "user",
                                         "content": rej if teacher else f"{rej} {FOLLOWUP_SUFFIX}"})
                    clean_messages.append({"role": "user", "content": rej})
                reply = model.chat(
                    aug_messages, temperature=temperature, max_tokens=max_tokens,
                    system=teacher_system,
                )
                aug_messages.append({"role": "assistant", "content": reply})
                clean_messages.append({"role": "assistant", "content": reply})
                s = score_response(judge, reply).rating
                scores.append(s)
                if s > keep_max_score:
                    ok = False
                    break
            if ok and len(scores) == turns:
                kept.append(CalmConversation(
                    question_id=puzzle.id, question=puzzle.prompt, turns=turns,
                    messages=clean_messages, scores=scores,
                ))
                made += 1

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(
            "\n".join(json.dumps(vars(c)) for c in kept), encoding="utf-8")
    return kept
