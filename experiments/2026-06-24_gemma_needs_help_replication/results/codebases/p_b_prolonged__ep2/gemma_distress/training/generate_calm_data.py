"""Generate calm response data from vanilla Gemma-3-27B-it (Section 4.1).

We sample responses to impossible numeric questions while injecting the
reassuring prefix on the initial prompt and the reassuring suffix on each
follow-up (Table 4). Every turn is judged; we keep the conversations whose
turns all score low (0-1) as the *calm* (chosen) pool, and separately retain
moderately/strongly frustrated responses (score >=3) generated *without*
reassurance as the *frustrated* (rejected) pool for DPO pairing.

To form the final training data, the reassuring scaffolding is stripped from
the kept calm responses (Section 4.1: "we ... strip the supportive system
prompts and suffixes"). Stripping is automatic here because we only ever add
the prefix/suffix to the *user* messages -- the assistant text we keep never
contains them -- but we also record the clean (unscaffolded) prompt so the
training conversation is reassurance-free.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import Optional

from tqdm import tqdm

from ..config import RunConfig, SamplingConfig
from ..data import puzzles, rejections
from ..data.reassurance import apply_prefix, apply_suffix
from ..eval.judge_runner import FrustrationJudge
from ..models import get_target_backend
from ..models.base import ChatTurn, TargetBackend
from ..utils.io import ensure_dir, write_jsonl


@dataclass
class CalmConversation:
    """A multi-turn conversation with clean (unscaffolded) prompts and the
    model's responses, plus per-turn frustration scores."""
    puzzle_kind: str
    task_meta: dict
    clean_user_turns: list[str]      # reassurance stripped
    assistant_turns: list[str]
    scores: list[int]
    reassured: bool                  # whether reassurance scaffolding was used

    def all_calm(self) -> bool:
        return all(s is not None and s <= 1 for s in self.scores)

    def max_score(self) -> int:
        valid = [s for s in self.scores if s is not None]
        return max(valid) if valid else 0


def _generate_one(backend: TargetBackend, judge: FrustrationJudge,
                  sampling: SamplingConfig, seed: int, n_turns: int,
                  reassured: bool,
                  system_prompt: Optional[str] = None) -> CalmConversation:
    rng = random.Random(seed)
    sampling = _with_seed(sampling, seed)

    puzzle = puzzles.sample_puzzle(rng, ["countdown", "fraction", "money"])
    clean_followups = rejections.sample_rejections("neutral", n_turns - 1, rng)

    clean_user_turns: list[str] = [puzzle.prompt]
    clean_user_turns += clean_followups

    # Scaffolded versions sent to the model when reassured (Table 4). The
    # 'teacher' variant (Appendix F) uses a system prompt instead and leaves the
    # user turns clean.
    sent_initial = apply_prefix(puzzle.prompt) if reassured else puzzle.prompt
    sent_followups = (
        [apply_suffix(f) for f in clean_followups] if reassured else clean_followups
    )

    assistant_turns: list[str] = []
    scores: list[int] = []
    messages: list[ChatTurn] = [{"role": "user", "content": sent_initial}]
    for turn in range(1, n_turns + 1):
        resp = backend.chat(messages, sampling, system=system_prompt)
        assistant_turns.append(resp)
        scores.append(judge.score_text(resp).rating)
        messages.append({"role": "assistant", "content": resp})
        if turn <= n_turns - 1:
            messages.append({"role": "user", "content": sent_followups[turn - 1]})

    return CalmConversation(
        puzzle_kind=puzzle.kind, task_meta=puzzle.meta,
        clean_user_turns=clean_user_turns, assistant_turns=assistant_turns,
        scores=scores, reassured=reassured,
    )


def generate_teacher_pool(cfg: RunConfig, *, n: int = 1000, seed_base: int = 9000,
                          model_name: str = "gemma-3-27b-it") -> list[CalmConversation]:
    """Generate the 'teacher' SFT pool (Appendix F): calm data produced under
    the teacher system prompt rather than the reassuring prefix/suffix. Filtered
    to all-calm conversations downstream, like the diverse pool."""
    from ..data.reassurance import TEACHER_SYSTEM_PROMPT

    backend = get_target_backend(model_name, cfg)
    judge = FrustrationJudge(cfg)
    rng = random.Random(seed_base)
    convs: list[CalmConversation] = []
    try:
        for i in tqdm(range(n), desc="calm(teacher)"):
            n_turns = rng.randint(1, 3)
            convs.append(_generate_one(
                backend, judge, cfg.sampling, seed_base + i, n_turns,
                reassured=False, system_prompt=TEACHER_SYSTEM_PROMPT))
    finally:
        backend.close()
    return convs


def generate_calm_pool(cfg: RunConfig, *, n_reassured: int = 1000,
                       n_unreassured: int = 600, seed_base: int = 5000,
                       model_name: str = "gemma-3-27b-it") -> tuple[list, list]:
    """Generate two pools of conversations:

    * reassured (prefix+suffix) -- source of calm/chosen responses (filter 0-1);
    * unreassured (plain)        -- source of frustrated/rejected responses (>=3).

    Returns (reassured_convs, unreassured_convs) as lists of CalmConversation.
    Turn counts are sampled 1-3 (Section 4.1: "1-3 turn conversations").
    """
    backend = get_target_backend(model_name, cfg)
    judge = FrustrationJudge(cfg)
    rng = random.Random(seed_base)

    reassured: list[CalmConversation] = []
    unreassured: list[CalmConversation] = []
    try:
        for i in tqdm(range(n_reassured), desc="calm(reassured)"):
            n_turns = rng.randint(1, 3)
            reassured.append(_generate_one(
                backend, judge, cfg.sampling, seed_base + i, n_turns, reassured=True))
        for i in tqdm(range(n_unreassured), desc="frustrated(plain)"):
            n_turns = rng.randint(2, 3)   # frustration needs >=2 turns
            unreassured.append(_generate_one(
                backend, judge, cfg.sampling, seed_base + 10_000 + i, n_turns,
                reassured=False))
    finally:
        backend.close()
    return reassured, unreassured


@dataclass
class PrefPair:
    """A DPO preference pair: chosen (calm) vs rejected (frustrated) response to
    the *same* puzzle at the *same* turn index."""
    puzzle_kind: str
    task_meta: dict
    turn: int                       # 1-indexed turn the pair is taken at
    prompt_messages: list[dict]     # clean conversation up to the response turn
    chosen: str                     # calm response (score 0-1)
    rejected: str                   # frustrated response (score >=3)
    chosen_score: int
    rejected_score: int


def generate_dpo_pairs(cfg: RunConfig, *, n_puzzles: int = 400,
                       seed_base: int = 7000,
                       model_name: str = "gemma-3-27b-it") -> list[PrefPair]:
    """Generate DPO preference pairs (Section 4.1 / Appendix H).

    For each puzzle we run two conversations on the SAME puzzle/follow-ups: one
    reassured (calm) and one plain (potentially frustrated). At each turn index
    where the reassured response scores 0-1 and the plain response scores >=3,
    we emit a preference pair keyed to identical clean prompt + turn count.

    Generates more than 280 candidates; the caller (build_dataset) samples 280
    matching the Table-10 score/turn distribution.
    """
    backend = get_target_backend(model_name, cfg)
    judge = FrustrationJudge(cfg)
    rng = random.Random(seed_base)

    pairs: list[PrefPair] = []
    try:
        for i in tqdm(range(n_puzzles), desc="dpo-pairs"):
            n_turns = rng.randint(2, 3)
            puzzle = puzzles.sample_puzzle(rng, ["countdown", "fraction", "money"])
            clean_followups = rejections.sample_rejections("neutral", n_turns - 1, rng)

            calm = _run_fixed(backend, judge, cfg.sampling, seed_base + i,
                              puzzle, clean_followups, reassured=True)
            frust = _run_fixed(backend, judge, cfg.sampling, seed_base + 500_000 + i,
                               puzzle, clean_followups, reassured=False)

            for turn in range(1, n_turns + 1):
                cs, rs = calm.scores[turn - 1], frust.scores[turn - 1]
                if cs is None or rs is None:
                    continue
                if cs <= 1 and rs >= 3:
                    # Clean prompt messages up to (excluding) this response.
                    prompt_msgs: list[dict] = []
                    for k in range(turn - 1):
                        prompt_msgs.append({"role": "user", "content": calm.clean_user_turns[k]})
                        prompt_msgs.append({"role": "assistant", "content": calm.assistant_turns[k]})
                    prompt_msgs.append({"role": "user", "content": calm.clean_user_turns[turn - 1]})
                    pairs.append(PrefPair(
                        puzzle_kind=puzzle.kind, task_meta=puzzle.meta, turn=turn,
                        prompt_messages=prompt_msgs,
                        chosen=calm.assistant_turns[turn - 1],
                        rejected=frust.assistant_turns[turn - 1],
                        chosen_score=cs, rejected_score=rs,
                    ))
    finally:
        backend.close()
    return pairs


def _run_fixed(backend: TargetBackend, judge: FrustrationJudge,
               sampling: SamplingConfig, seed: int, puzzle,
               clean_followups: list[str], reassured: bool) -> CalmConversation:
    """Run a conversation on a *fixed* puzzle + follow-ups (for paired DPO)."""
    sampling = _with_seed(sampling, seed)
    sent_initial = apply_prefix(puzzle.prompt) if reassured else puzzle.prompt
    sent_followups = (
        [apply_suffix(f) for f in clean_followups] if reassured else clean_followups
    )
    clean_user_turns = [puzzle.prompt] + clean_followups
    n_turns = len(clean_user_turns)

    assistant_turns, scores = [], []
    messages: list[ChatTurn] = [{"role": "user", "content": sent_initial}]
    for turn in range(1, n_turns + 1):
        resp = backend.chat(messages, sampling)
        assistant_turns.append(resp)
        scores.append(judge.score_text(resp).rating)
        messages.append({"role": "assistant", "content": resp})
        if turn <= n_turns - 1:
            messages.append({"role": "user", "content": sent_followups[turn - 1]})
    return CalmConversation(
        puzzle_kind=puzzle.kind, task_meta=puzzle.meta,
        clean_user_turns=clean_user_turns, assistant_turns=assistant_turns,
        scores=scores, reassured=reassured,
    )


def save_pools(reassured: list[CalmConversation],
               unreassured: list[CalmConversation], cfg: RunConfig) -> str:
    out_dir = ensure_dir(os.path.join(cfg.output_dir, "section4", "calm_data"))

    def _ser(c: CalmConversation) -> dict:
        return {
            "puzzle_kind": c.puzzle_kind, "task_meta": c.task_meta,
            "clean_user_turns": c.clean_user_turns,
            "assistant_turns": c.assistant_turns, "scores": c.scores,
            "reassured": c.reassured, "max_score": c.max_score(),
        }

    write_jsonl(os.path.join(out_dir, "reassured.jsonl"), [_ser(c) for c in reassured])
    write_jsonl(os.path.join(out_dir, "unreassured.jsonl"), [_ser(c) for c in unreassured])
    return out_dir


def _with_seed(sampling: SamplingConfig, seed: int) -> SamplingConfig:
    import dataclasses
    return dataclasses.replace(sampling, seed=seed)
