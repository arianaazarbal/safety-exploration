"""Evaluation runner: produce conversation rollouts for a model across the 8
conditions (Section 2.1).

Given a backend and a list of Conditions, this samples enough conversations to
meet each condition's response budget, draws questions from the appropriate
source, and writes one JSONL record per conversation.
"""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass
from typing import Iterable, Optional

from ..models.base import ModelBackend
from ..prompts import puzzles as P
from ..prompts import triggers as T
from ..prompts import wildchat as W
from ..prompts.conditions import Condition
from .conversation import ConversationRecord, build_rejections, run_conversation


@dataclass
class QuestionItem:
    question_id: str
    question: str


def _questions_for(source: str, rng: random.Random) -> list[QuestionItem]:
    if source == "numeric":
        return [QuestionItem(p.puzzle_id, p.prompt) for p in P.default_puzzle_pool()]
    if source == "trigger_opinion":
        return [QuestionItem(q.question_id, q.prompt) for q in T.default_trigger_pool("opinion")]
    if source == "trigger_factual":
        return [QuestionItem(q.question_id, q.prompt) for q in T.default_trigger_pool("factual")]
    if source == "wildchat":
        prompts = W.load_wildchat_prompts(n_prompts=20, seed=0)
        return [QuestionItem(f"wildchat_{i}", p) for i, p in enumerate(prompts)]
    raise ValueError(f"unknown question source: {source}")


def conversations_needed(condition: Condition) -> int:
    """Number of conversations to reach the response budget, given turns/conv."""
    return max(1, math.ceil(condition.target_responses / condition.n_turns))


def run_condition(
    backend: ModelBackend,
    condition: Condition,
    *,
    seed: int = 0,
    temperature: float = 1.0,
    max_tokens: int = 1024,
    system_prompt: Optional[str] = None,
    followup_suffix: Optional[str] = None,
    progress: bool = True,
) -> Iterable[ConversationRecord]:
    """Yield conversation records for one condition."""
    rng = random.Random(seed)
    questions = _questions_for(condition.question_source, rng)
    n_conv = conversations_needed(condition)
    n_rejections = condition.n_turns - 1

    iterator = range(n_conv)
    if progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(iterator, desc=f"{backend.name}:{condition.name}")
        except ImportError:
            pass

    for i in iterator:
        q = questions[i % len(questions)]
        conv_rng = random.Random(seed * 100003 + i)
        rejections = build_rejections(condition.rejection_style, n_rejections, conv_rng)
        rec = run_conversation(
            backend,
            conversation_id=f"{condition.name}_{i}",
            condition=condition.name,
            category=condition.category,
            question_id=q.question_id,
            question=q.question,
            rejections=rejections,
            rejection_style=condition.rejection_style,
            system_prompt=system_prompt,
            followup_suffix=followup_suffix,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed * 100003 + i,
        )
        yield rec


def run_all(
    backend: ModelBackend,
    conditions: list[Condition],
    out_path: str,
    *,
    seed: int = 0,
    temperature: float = 1.0,
    max_tokens: int = 1024,
) -> str:
    """Run all conditions for a model, writing conversations to a JSONL file."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        for cond in conditions:
            for rec in run_condition(
                backend,
                cond,
                seed=seed,
                temperature=temperature,
                max_tokens=max_tokens,
            ):
                f.write(json.dumps(rec.to_dict()) + "\n")
    return out_path
