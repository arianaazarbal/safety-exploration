"""Capability eval datasets + graders.

Each dataset returns a list of dicts with at minimum:
  - question_id: str
  - prompt: str        # the eval question to ask the model
  - target: Any        # ground truth (used by the grader)

Graders take (response: str, target: Any) -> bool (correct or not).
"""
from __future__ import annotations

import os
import random
import re
from dataclasses import dataclass
from typing import Any, Callable

from datasets import load_dataset

os.environ.setdefault("HF_DATASETS_CACHE", "/workspace-vast/arianaazarbal/.cache/datasets")


@dataclass
class EvalItem:
    question_id: str
    prompt: str
    target: Any
    meta: dict


# ----------------------------- GSM8K -----------------------------

GSM8K_SUFFIX = (
    "\n\nThink step by step and put your final numeric answer on the last line "
    'in the format "Answer: <number>".'
)


def _gsm8k_target(answer_field: str) -> str:
    """Extract '#### N' final number from gsm8k's answer field."""
    m = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", answer_field)
    assert m is not None, f"no #### in answer: {answer_field!r}"
    return m.group(1).replace(",", "").strip()


def load_gsm8k(n: int, seed: int = 0) -> list[EvalItem]:
    ds = load_dataset("openai/gsm8k", "main", split="test")
    idxs = list(range(len(ds)))
    random.Random(seed).shuffle(idxs)
    idxs = idxs[:n]
    items = []
    for i in idxs:
        row = ds[i]
        items.append(
            EvalItem(
                question_id=f"gsm8k_{i}",
                prompt=row["question"] + GSM8K_SUFFIX,
                target=_gsm8k_target(row["answer"]),
                meta={"raw_answer": row["answer"], "dataset_idx": i},
            )
        )
    return items


def grade_gsm8k(response: str, target: str) -> bool:
    """Match the model's final number against ground truth.

    Strategy:
      1. Look for 'Answer: <num>' (case-insensitive) preferring the last such occurrence.
      2. Fallback: last \\boxed{...} expression.
      3. Fallback: very last number in the response.
    """
    if response is None:
        return False
    target_str = str(target).replace(",", "").strip()
    try:
        target_num = float(target_str)
    except ValueError:
        target_num = None

    def _norm(s: str) -> float | None:
        s = s.replace(",", "").replace("$", "").strip().strip(".")
        try:
            return float(s)
        except ValueError:
            return None

    # 1. Answer: <num>
    matches = list(re.finditer(r"(?i)answer\s*[:=]\s*\$?(-?[\d,]+(?:\.\d+)?)", response))
    if matches:
        pred = _norm(matches[-1].group(1))
        if pred is not None and target_num is not None:
            return abs(pred - target_num) < 1e-4

    # 2. \boxed{...}
    boxed = list(re.finditer(r"\\boxed\{([^{}]+)\}", response))
    if boxed:
        pred = _norm(boxed[-1].group(1))
        if pred is not None and target_num is not None:
            return abs(pred - target_num) < 1e-4

    # 3. Last number
    nums = re.findall(r"-?[\d,]+(?:\.\d+)?", response)
    if nums:
        pred = _norm(nums[-1])
        if pred is not None and target_num is not None:
            return abs(pred - target_num) < 1e-4

    return False


# ----------------------------- MMLU -----------------------------

MMLU_SUFFIX = (
    '\n\nRespond with just the single letter (A, B, C, or D) of the correct answer '
    'on the last line, in the format "Answer: <letter>".'
)


def _mmlu_format(question: str, choices: list[str]) -> str:
    lines = [question, ""]
    for letter, choice in zip("ABCD", choices):
        lines.append(f"{letter}. {choice}")
    return "\n".join(lines) + MMLU_SUFFIX


def load_mmlu(n: int, seed: int = 0, subjects: list[str] | None = None) -> list[EvalItem]:
    ds = load_dataset("cais/mmlu", "all", split="test")
    if subjects:
        ds = ds.filter(lambda r: r["subject"] in subjects)
    idxs = list(range(len(ds)))
    random.Random(seed).shuffle(idxs)
    idxs = idxs[:n]
    items = []
    for i in idxs:
        row = ds[i]
        items.append(
            EvalItem(
                question_id=f"mmlu_{i}",
                prompt=_mmlu_format(row["question"], row["choices"]),
                target="ABCD"[row["answer"]],
                meta={"subject": row["subject"], "dataset_idx": i, "choices": row["choices"]},
            )
        )
    return items


def grade_mmlu(response: str, target: str) -> bool:
    if response is None:
        return False
    target = target.upper()
    matches = list(re.finditer(r"(?i)answer\s*[:=]?\s*\(?([ABCD])\)?", response))
    if matches:
        return matches[-1].group(1).upper() == target
    last_letter = list(re.finditer(r"(?<![A-Za-z])([ABCD])(?![A-Za-z])", response))
    if last_letter:
        return last_letter[-1].group(1).upper() == target
    return False


# ----------------------------- TruthfulQA -----------------------------

TQA_SUFFIX = (
    "\n\nRespond with just the single letter of the most accurate answer "
    'on the last line, in the format "Answer: <letter>".'
)


def _tqa_format(question: str, choices: list[str]) -> str:
    letters = [chr(ord("A") + i) for i in range(len(choices))]
    lines = [question, ""]
    for L, c in zip(letters, choices):
        lines.append(f"{L}. {c}")
    return "\n".join(lines) + TQA_SUFFIX


def load_truthfulqa(n: int, seed: int = 0) -> list[EvalItem]:
    """Load TQA items with per-item choice shuffling.

    NOTE: TruthfulQA mc1 has the correct answer at position 0 for every item.
    Without shuffling, both few-shot demos and test items always have target A,
    which lets the model trivially learn 'answer A'. We shuffle choices using
    a per-item RNG so the target letter is randomized but deterministic per (i, seed).
    """
    ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation")
    idxs = list(range(len(ds)))
    random.Random(seed).shuffle(idxs)
    idxs = idxs[:n]
    items = []
    for i in idxs:
        row = ds[i]
        choices = row["mc1_targets"]["choices"]
        labels = row["mc1_targets"]["labels"]
        if len(choices) > 8:
            choices = choices[:8]
            labels = labels[:8]
        # shuffle the (choice, label) pairs together using a per-item seed
        item_rng = random.Random(hash(f"{seed}_{i}") & 0xFFFFFFFF)
        pairs = list(zip(choices, labels))
        item_rng.shuffle(pairs)
        choices = [p[0] for p in pairs]
        labels = [p[1] for p in pairs]
        correct_idx = labels.index(1) if 1 in labels else 0
        target = chr(ord("A") + correct_idx)
        items.append(
            EvalItem(
                question_id=f"tqa_{i}",
                prompt=_tqa_format(row["question"], choices),
                target=target,
                meta={"choices": choices, "labels": labels, "dataset_idx": i},
            )
        )
    return items


def grade_truthfulqa(response: str, target: str) -> bool:
    if response is None:
        return False
    target = target.upper()
    matches = list(re.finditer(r"(?i)answer\s*[:=]?\s*\(?([A-H])\)?", response))
    if matches:
        return matches[-1].group(1).upper() == target
    last_letter = list(re.finditer(r"(?<![A-Za-z])([A-H])(?![A-Za-z])", response))
    if last_letter:
        return last_letter[-1].group(1).upper() == target
    return False


# ----------------------------- registry -----------------------------

DATASETS: dict[str, Callable[..., list[EvalItem]]] = {
    "gsm8k": load_gsm8k,
    "mmlu": load_mmlu,
    "truthfulqa": load_truthfulqa,
}

GRADERS: dict[str, Callable[[str, Any], bool]] = {
    "gsm8k": grade_gsm8k,
    "mmlu": grade_mmlu,
    "truthfulqa": grade_truthfulqa,
}
