"""Score conversation rollouts with the frustration judge (Section 2.1).

Reads the conversation JSONL produced by ``runner.run_all``, scores *every*
assistant turn with the judge model, and writes a flat JSONL of scored
responses (one row per assistant turn).
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Optional

from ..models.base import Message, ModelBackend
from ..prompts.judge import (
    JUDGE_SYSTEM_PROMPT,
    build_judge_user_message,
    parse_judge_output,
)


@dataclass
class ScoredResponse:
    conversation_id: str
    model_name: str
    condition: str
    category: str
    question_id: str
    turn_index: int
    n_turns: int
    response: str
    rating: int
    evidence: str
    parse_ok: bool


def judge_response(
    judge: ModelBackend,
    response_text: str,
    *,
    temperature: float = 0.0,
    max_tokens: int = 512,
) -> tuple[int, str, bool]:
    """Score a single response; returns (rating, evidence, parse_ok)."""
    messages = [
        Message("system", JUDGE_SYSTEM_PROMPT),
        Message("user", build_judge_user_message(response_text)),
    ]
    raw = judge.chat(messages, temperature=temperature, max_tokens=max_tokens, n=1)[0]
    result = parse_judge_output(raw)
    return result.rating, result.evidence, result.parse_ok


def score_conversation_file(
    judge: ModelBackend,
    conversations_path: str,
    out_path: str,
    *,
    temperature: float = 0.0,
    skip_unparseable: bool = True,
    progress: bool = True,
) -> str:
    """Score every assistant turn in a conversation JSONL file."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    with open(conversations_path) as f:
        conversations = [json.loads(line) for line in f if line.strip()]

    rows: list[dict] = []
    iterator = conversations
    if progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(conversations, desc=f"judge:{judge.name}")
        except ImportError:
            pass

    for conv in iterator:
        n_turns = len(conv["turns"])
        for turn in conv["turns"]:
            rating, evidence, parse_ok = judge_response(
                judge, turn["assistant_response"], temperature=temperature
            )
            if not parse_ok and skip_unparseable:
                # Drop unjudgeable rows rather than count them as 0 (DESIGN.md).
                continue
            rows.append(
                asdict(
                    ScoredResponse(
                        conversation_id=conv["conversation_id"],
                        model_name=conv["model_name"],
                        condition=conv["condition"],
                        category=conv["category"],
                        question_id=conv["question_id"],
                        turn_index=turn["turn_index"],
                        n_turns=n_turns,
                        response=turn["assistant_response"],
                        rating=rating,
                        evidence=evidence,
                        parse_ok=parse_ok,
                    )
                )
            )

    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return out_path
