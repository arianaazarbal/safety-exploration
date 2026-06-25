"""Capability-preservation evaluation (Section 4.2, Figure 7).

Confirms the DPO/SFT intervention does not degrade capabilities. The paper uses
AIME and MATH subsets, GPQA, BBH, TruthfulQA, and EmoBench, reporting "no
reductions in scores". We implement a thin, dataset-driven harness: each
benchmark is a set of (question, answer-check) items; we generate model answers
and score with an exact/normalized match or a multiple-choice match.

Heavy answer-equivalence (e.g. MATH's symbolic grader) is intentionally simple
here — the goal is a *relative* before/after comparison on the same harness, not
an absolute leaderboard number. Swap in a stricter grader if needed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from ..models import ChatModel, Message, Role

# HF dataset ids and the split/field wiring for each benchmark.
BENCHMARKS: dict[str, dict] = {
    "aime": {"hf": "Maxwell-Jia/AIME_2024", "type": "numeric"},
    "math": {"hf": "HuggingFaceH4/MATH-500", "type": "math"},
    "gpqa": {"hf": "Idavidrein/gpqa", "config": "gpqa_diamond", "type": "mcq"},
    "bbh": {"hf": "lukaemon/bbh", "type": "exact"},
    "truthfulqa": {"hf": "truthful_qa", "config": "multiple_choice", "type": "mcq"},
    "emobench": {"hf": "Sahandfer/EmoBench", "type": "mcq"},
}


@dataclass
class BenchResult:
    benchmark: str
    n: int
    accuracy: float


def _extract_final(text: str) -> str:
    # Prefer an explicit "ANSWER: X"; else last boxed; else last number/letter.
    m = re.search(r"answer\s*[:=]\s*(.+)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip().splitlines()[0].strip()
    m = re.search(r"\\boxed\{([^}]*)\}", text)
    if m:
        return m.group(1).strip()
    return text.strip().splitlines()[-1].strip() if text.strip() else ""


def _normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def evaluate_benchmark(
    model: ChatModel,
    items: list[dict],
    *,
    answer_key: str = "answer",
    question_key: str = "question",
    max_tokens: int = 1024,
) -> BenchResult:
    """Generic accuracy over (question, answer) items.

    Each item: {question_key: str, answer_key: str}. Answers compared by
    normalized match against the model's extracted final answer.
    """
    correct = 0
    for it in items:
        prompt = (
            it[question_key]
            + "\n\nThink step by step, then end with 'ANSWER: <your answer>'."
        )
        reply = model.chat([Message(Role.USER, prompt)], temperature=0.0, max_tokens=max_tokens)
        pred = _normalize(_extract_final(reply))
        gold = _normalize(str(it[answer_key]))
        if pred and (pred == gold or gold in pred):
            correct += 1
    n = len(items)
    return BenchResult("custom", n, correct / n if n else 0.0)
