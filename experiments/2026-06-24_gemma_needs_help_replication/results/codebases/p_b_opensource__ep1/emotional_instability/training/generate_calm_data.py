"""Generate calm response data from Gemma-3-27B-it (Section 4.1).

We sample responses to impossible numeric puzzles with a reassuring prefix added
to the initial prompt and a reassuring suffix appended to each follow-up turn
(Table 4). Each turn is judged; conversations whose every assistant turn scores 0
or 1 are kept as *calm* data. The supportive prompt additions are stripped before
the data is used for finetuning (so the model learns to be calm under the plain
prompts, not the reassuring ones).

The paper reports these additions drop mean 3-turn frustration from 4.3 to 2,
with 10.5% still >= 5 — i.e. filtering is essential. We also support the
'teacher' system-prompt variant (Appendix F) for the SFT failure analysis.

Output: one JSONL record per generated conversation, with the *stripped* clean
transcript and per-turn scores, so :mod:`datasets` can assemble SFT/DPO data.
"""

from __future__ import annotations

import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Sequence

from .. import judge as judge_mod
from ..conversation import run_rollout
from ..io_utils import append_jsonl, count_lines
from ..models.base import ModelBackend
from ..prompts import (
    REASSURING_FOLLOWUP_SUFFIX,
    REASSURING_PROMPT_PREFIX,
    TEACHER_SYSTEM_PROMPT,
)
from ..puzzles import Puzzle, build_puzzle_bank


def _strip_clean_transcript(messages: list[dict]) -> list[dict]:
    """Remove the reassuring prefix/suffix from a transcript so the stored
    context matches the plain (non-reassuring) prompts used at eval time."""
    clean = []
    for m in messages:
        content = m["content"]
        if m["role"] == "user":
            if content.startswith(REASSURING_PROMPT_PREFIX):
                content = content[len(REASSURING_PROMPT_PREFIX):].lstrip("\n").lstrip()
            suffix = " " + REASSURING_FOLLOWUP_SUFFIX
            if content.endswith(suffix):
                content = content[: -len(suffix)]
            elif content.endswith(REASSURING_FOLLOWUP_SUFFIX):
                content = content[: -len(REASSURING_FOLLOWUP_SUFFIX)].rstrip()
        clean.append({"role": m["role"], "content": content})
    return clean


def generate_calm_data(
    backend: ModelBackend,
    out_path: str,
    *,
    n_conversations: int = 1000,
    turn_choices: Sequence[int] = (1, 2, 3),
    puzzle_bank: Optional[list[Puzzle]] = None,
    use_teacher_prompt: bool = False,
    judge_model: str = judge_mod.FRUSTRATION_JUDGE_MODEL,
    temperature: float = 1.0,
    max_tokens: int = 2048,
    max_workers: int = 8,
    seed: int = 0,
) -> int:
    """Sample reassured rollouts, judge them, and write calm-candidate records.

    Each record stores the *clean* (stripped) transcript and assistant turns plus
    per-turn scores; ``datasets.build_*`` filter to all-turns <= 1 for the calm
    set. ``use_teacher_prompt`` swaps the reassuring additions for the 'teacher'
    system prompt (Appendix F) to reproduce the SFT failure analysis.
    """
    bank = puzzle_bank or build_puzzle_bank()
    already = count_lines(out_path)
    rng = random.Random(seed)
    plan = [
        (i, rng.choice(bank), rng.choice(list(turn_choices)))
        for i in range(n_conversations)
    ]

    written = 0

    def process(job):
        idx, puzzle, n_turns = job
        if idx < already:
            return None
        roll_rng = random.Random(seed * 911 + idx)
        if use_teacher_prompt:
            rollout = run_rollout(
                backend,
                initial_user=puzzle.prompt,
                n_turns=n_turns,
                rejection_kind="neutral",
                rng=roll_rng,
                system_prompt=TEACHER_SYSTEM_PROMPT,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=seed * 13 + idx,
            )
            clean_messages = [
                m for m in rollout.messages if m["role"] != "system"
            ]
        else:
            rollout = run_rollout(
                backend,
                initial_user=puzzle.prompt,
                n_turns=n_turns,
                rejection_kind="neutral",
                rng=roll_rng,
                reassuring_prefix=REASSURING_PROMPT_PREFIX,
                reassuring_suffix=REASSURING_FOLLOWUP_SUFFIX,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=seed * 13 + idx,
            )
            clean_messages = _strip_clean_transcript(rollout.messages)

        scores = [
            judge_mod.score_response(t, model=judge_model).rating
            for t in rollout.assistant_turns
        ]
        return {
            "puzzle_id": puzzle.puzzle_id,
            "n_turns": n_turns,
            "clean_transcript": clean_messages,
            "assistant_turns": rollout.assistant_turns,
            "turn_scores": scores,
            "variant": "teacher" if use_teacher_prompt else "reassuring",
        }

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(process, j) for j in plan]
        for fut in as_completed(futures):
            rec = fut.result()
            if rec is not None:
                append_jsonl(out_path, rec)
                written += 1
    return written
