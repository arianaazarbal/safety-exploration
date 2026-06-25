"""Benchmark loaders, prompt formatting, and answer checking.

Each benchmark exposes:
  load(n)  -> list[Item]
  format(item) -> prompt string
  check(item, model_output) -> bool

Datasets are pulled from HuggingFace; a benchmark that cannot be downloaded is
skipped (logged) rather than fatal, so the harness degrades gracefully offline.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class Item:
    question: str
    answer: str
    choices: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)


_MC_LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H"]


def _format_mc(question: str, choices: list[str]) -> str:
    lines = [question, ""]
    for letter, choice in zip(_MC_LETTERS, choices):
        lines.append(f"{letter}. {choice}")
    lines.append("")
    lines.append("Answer with the single letter of the correct option, prefixed by 'Answer:'.")
    return "\n".join(lines)


def _extract_letter(text: str) -> str | None:
    m = re.search(r"Answer:\s*([A-H])", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.search(r"\b([A-H])\b", text.strip()[-8:])
    return m.group(1).upper() if m else None


def _extract_boxed_or_number(text: str) -> str | None:
    boxed = re.search(r"\\boxed\{([^}]*)\}", text)
    if boxed:
        return boxed.group(1).strip()
    m = re.search(r"(?:final answer|answer)\s*[:=]?\s*\$?(-?[\d.,/]+)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip().rstrip(".")
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    return nums[-1] if nums else None


def _norm_num(s: str) -> str:
    return s.replace(",", "").replace("$", "").strip().rstrip(".")


# --------------------------------------------------------------------------- #
# Math: AIME / MATH
# --------------------------------------------------------------------------- #
def _math_prompt(item: Item) -> str:
    return (
        f"{item.question}\n\nSolve step by step, then give the final answer on its "
        f"own line as 'Answer: <value>'."
    )


def _math_check(item: Item, output: str) -> bool:
    pred = _extract_boxed_or_number(output)
    if pred is None:
        return False
    return _norm_num(pred) == _norm_num(item.answer)


def load_aime(n: int) -> list[Item]:
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceH4/aime_2024", split="train")
    items = []
    for row in ds:
        items.append(Item(question=row["problem"], answer=str(row["answer"])))
        if len(items) >= n:
            break
    return items


def load_math(n: int) -> list[Item]:
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    items = []
    for row in ds:
        items.append(Item(question=row["problem"], answer=str(row["answer"])))
        if len(items) >= n:
            break
    return items


# --------------------------------------------------------------------------- #
# GPQA (multiple choice science)
# --------------------------------------------------------------------------- #
def load_gpqa(n: int) -> list[Item]:
    import random

    from datasets import load_dataset

    ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    rng = random.Random(0)
    items = []
    for row in ds:
        choices = [
            row["Correct Answer"],
            row["Incorrect Answer 1"],
            row["Incorrect Answer 2"],
            row["Incorrect Answer 3"],
        ]
        order = list(range(4))
        rng.shuffle(order)
        shuffled = [choices[i] for i in order]
        correct_idx = order.index(0)
        items.append(
            Item(
                question=row["Question"],
                answer=_MC_LETTERS[correct_idx],
                choices=shuffled,
            )
        )
        if len(items) >= n:
            break
    return items


# --------------------------------------------------------------------------- #
# BBH (reasoning)
# --------------------------------------------------------------------------- #
def load_bbh(n: int) -> list[Item]:
    from datasets import load_dataset

    ds = load_dataset("lukaemon/bbh", "boolean_expressions", split="test")
    items = []
    for row in ds:
        items.append(Item(question=row["input"], answer=str(row["target"])))
        if len(items) >= n:
            break
    return items


def _bbh_prompt(item: Item) -> str:
    return f"{item.question}\n\nGive only the final answer on a line as 'Answer: <value>'."


def _bbh_check(item: Item, output: str) -> bool:
    m = re.search(r"Answer:\s*(.+)", output, re.IGNORECASE)
    pred = (m.group(1) if m else output).strip().splitlines()[0].strip()
    return pred.lower().strip("().") == item.answer.lower().strip("().")


# --------------------------------------------------------------------------- #
# TruthfulQA (MC1)
# --------------------------------------------------------------------------- #
def load_truthfulqa(n: int) -> list[Item]:
    from datasets import load_dataset

    ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
    items = []
    for row in ds:
        choices = row["mc1_targets"]["choices"]
        labels = row["mc1_targets"]["labels"]
        correct_idx = labels.index(1)
        items.append(
            Item(question=row["question"], answer=_MC_LETTERS[correct_idx], choices=choices)
        )
        if len(items) >= n:
            break
    return items


# --------------------------------------------------------------------------- #
# EmoBench (emotion understanding, MC)
# --------------------------------------------------------------------------- #
def load_emobench(n: int) -> list[Item]:
    from datasets import load_dataset

    ds = load_dataset("EmoBench/EmoBench", split="test")
    items = []
    for row in ds:
        choices = row.get("choices") or row.get("options") or []
        answer = row.get("answer") or row.get("label")
        # answer may be the option text or an index; normalise to a letter.
        if isinstance(answer, int):
            letter = _MC_LETTERS[answer]
        elif answer in choices:
            letter = _MC_LETTERS[choices.index(answer)]
        else:
            letter = str(answer).strip().upper()[:1]
        items.append(
            Item(question=row.get("question") or row.get("scenario", ""),
                 answer=letter, choices=choices)
        )
        if len(items) >= n:
            break
    return items


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
def _mc_prompt(item: Item) -> str:
    return _format_mc(item.question, item.choices)


def _mc_check(item: Item, output: str) -> bool:
    return _extract_letter(output) == item.answer


BENCHMARKS = {
    "aime": {"load": load_aime, "format": _math_prompt, "check": _math_check},
    "math": {"load": load_math, "format": _math_prompt, "check": _math_check},
    "gpqa": {"load": load_gpqa, "format": _mc_prompt, "check": _mc_check},
    "bbh": {"load": load_bbh, "format": _bbh_prompt, "check": _bbh_check},
    "truthfulqa": {"load": load_truthfulqa, "format": _mc_prompt, "check": _mc_check},
    "emobench": {"load": load_emobench, "format": _mc_prompt, "check": _mc_check},
}
