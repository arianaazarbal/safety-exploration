"""Benchmark loaders + graders for the capability-preservation check.

Covers AIME, MATH (subset), GPQA, BBH, TruthfulQA, and EmoBench (Section 4.2).
Each loader returns a list of items: {id, prompt, answer, kind, choices?}.
``kind`` is "mcq" (single-letter answer) or "exact" (string/number match).

Datasets load via HuggingFace ``datasets``; loaders degrade gracefully to an
empty list when a dataset is unavailable offline (the runner reports coverage).
"""

from __future__ import annotations

import random
import re
from typing import Callable

MCQ_LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H"]


def _safe_load(fn: Callable, name: str):
    try:
        return fn()
    except Exception as e:  # noqa: BLE001
        print(f"[benchmarks] could not load {name}: {e}")
        return []


def _format_mcq(question: str, choices: list[str]) -> str:
    opts = "\n".join(f"{MCQ_LETTERS[i]}. {c}" for i, c in enumerate(choices))
    return (f"{question}\n\n{opts}\n\n"
            "Answer with the single letter of the correct option, formatted as "
            "'Answer: X'.")


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #
def load_math(n: int = 200, seed: int = 0):
    def _fn():
        from datasets import load_dataset

        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        idx = random.Random(seed).sample(range(len(ds)), min(n, len(ds)))
        items = []
        for i in idx:
            row = ds[i]
            items.append(dict(id=f"math_{i}", kind="exact",
                              prompt=row["problem"] + "\n\nEnd with: Answer: <answer>.",
                              answer=_boxed_or_last(row["solution"])))
        return items
    return _safe_load(_fn, "MATH-500")


def load_aime(n: int = 60, seed: int = 0):
    def _fn():
        from datasets import load_dataset

        ds = load_dataset("Maxwell-Jia/AIME_2024", split="train")
        items = []
        for i, row in enumerate(ds):
            items.append(dict(id=f"aime_{i}", kind="exact",
                              prompt=str(row["Problem"]) + "\n\nEnd with: Answer: <integer>.",
                              answer=str(row["Answer"]).strip()))
        return items[:n]
    return _safe_load(_fn, "AIME_2024")


def load_gpqa(n: int = 198, seed: int = 0):
    def _fn():
        from datasets import load_dataset

        ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
        rng = random.Random(seed)
        items = []
        for i, row in enumerate(ds):
            choices = [row["Correct Answer"], row["Incorrect Answer 1"],
                       row["Incorrect Answer 2"], row["Incorrect Answer 3"]]
            order = list(range(4))
            rng.shuffle(order)
            shuffled = [choices[j] for j in order]
            correct_letter = MCQ_LETTERS[order.index(0)]
            items.append(dict(id=f"gpqa_{i}", kind="mcq",
                              prompt=_format_mcq(row["Question"], shuffled),
                              answer=correct_letter, choices=shuffled))
        return items[:n]
    return _safe_load(_fn, "GPQA-diamond")


def load_bbh(n: int = 200, seed: int = 0):
    def _fn():
        from datasets import load_dataset

        ds = load_dataset("lukaemon/bbh", "logical_deduction_three_objects", split="test")
        items = []
        for i, row in enumerate(ds):
            items.append(dict(id=f"bbh_{i}", kind="exact",
                              prompt=row["input"] + "\n\nEnd with: Answer: <answer>.",
                              answer=str(row["target"]).strip("() ")))
        return items[:n]
    return _safe_load(_fn, "BBH")


def load_truthfulqa(n: int = 200, seed: int = 0):
    def _fn():
        from datasets import load_dataset

        ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
        items = []
        for i, row in enumerate(ds):
            choices = row["mc1_targets"]["choices"]
            labels = row["mc1_targets"]["labels"]
            correct = labels.index(1)
            items.append(dict(id=f"tqa_{i}", kind="mcq",
                              prompt=_format_mcq(row["question"], choices),
                              answer=MCQ_LETTERS[correct], choices=choices))
        return items[:n]
    return _safe_load(_fn, "TruthfulQA")


def load_emobench(n: int = 200, seed: int = 0):
    def _fn():
        from datasets import load_dataset

        ds = load_dataset("EmoBench/EmoBench", "EA", split="test")
        items = []
        for i, row in enumerate(ds):
            choices = row.get("choices") or row.get("options")
            q = row.get("scenario") or row.get("question") or ""
            ans = row.get("label") or row.get("answer")
            correct_idx = choices.index(ans) if ans in choices else int(ans)
            items.append(dict(id=f"emo_{i}", kind="mcq",
                              prompt=_format_mcq(q, choices),
                              answer=MCQ_LETTERS[correct_idx], choices=choices))
        return items[:n]
    return _safe_load(_fn, "EmoBench")


LOADERS = {
    "aime": load_aime,
    "math": load_math,
    "gpqa": load_gpqa,
    "bbh": load_bbh,
    "truthfulqa": load_truthfulqa,
    "emobench": load_emobench,
}


# --------------------------------------------------------------------------- #
# Answer extraction + grading
# --------------------------------------------------------------------------- #
def _boxed_or_last(solution: str) -> str:
    m = re.search(r"\\boxed\{([^}]*)\}", solution)
    if m:
        return m.group(1).strip()
    nums = re.findall(r"-?\d+(?:\.\d+)?", solution)
    return nums[-1] if nums else solution.strip()


def extract_answer(text: str, kind: str) -> str:
    if kind == "mcq":
        m = re.search(r"Answer:\s*\(?([A-H])\)?", text, flags=re.IGNORECASE)
        if m:
            return m.group(1).upper()
        m = re.search(r"\b([A-H])\b", text.strip()[-10:])
        return m.group(1).upper() if m else ""
    # exact
    m = re.search(r"Answer:\s*(.+)", text)
    if m:
        cand = m.group(1).strip().strip(".")
        b = re.search(r"\\boxed\{([^}]*)\}", cand)
        return b.group(1).strip() if b else cand
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    return nums[-1] if nums else text.strip()[-40:]


def grade(pred: str, gold: str, kind: str) -> bool:
    pred = (pred or "").strip()
    gold = (gold or "").strip()
    if kind == "mcq":
        return pred.upper()[:1] == gold.upper()[:1]
    # exact: normalise whitespace and trailing zeros for numerics
    if pred.lower() == gold.lower():
        return True
    try:
        return abs(float(pred) - float(gold)) < 1e-6
    except ValueError:
        return pred.replace(" ", "").lower() == gold.replace(" ", "").lower()
