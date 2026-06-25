"""Benchmark loaders + answer extraction + scoring.

Covers the capability suite the paper uses to check the DPO/SFT finetunes do
not degrade capabilities (Section 4.2): AIME, MATH subset, GPQA, BBH,
TruthfulQA, and the emotion benchmark EmoBench.

Each benchmark exposes a uniform :class:`BenchmarkSpec` with:
  * ``load(n)``      -> list of {"prompt", "answer", "type"} items
  * ``score(pred, item)`` -> bool

Answer formats are normalised to either multiple-choice letters or
numeric/string final answers extracted from a ``The answer is X`` style tail.
Datasets are loaded from HuggingFace; if a dataset is unavailable the loader
returns an empty list so the suite degrades gracefully.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

import config

MCQ_INSTRUCTION = (
    "Answer the following multiple-choice question. End your response with a "
    "line of the form 'The answer is X' where X is the correct option letter.")
NUM_INSTRUCTION = (
    "Solve the following problem. End your response with a line of the form "
    "'The answer is X' where X is your final answer.")


@dataclass
class BenchmarkSpec:
    name: str
    load: Callable[[int], list[dict]]
    score: Callable[[str, dict], bool]


# --------------------------------------------------------------------------- #
# Answer extraction / matching
# --------------------------------------------------------------------------- #
def extract_final_answer(text: str) -> str:
    m = re.findall(r"answer is[:\s]*\$?([A-Za-z0-9\-\./]+)", text, re.IGNORECASE)
    if m:
        return m[-1].strip().rstrip(".")
    # fall back to last boxed or last number/letter
    boxed = re.findall(r"\\boxed\{([^}]*)\}", text)
    if boxed:
        return boxed[-1].strip()
    nums = re.findall(r"-?\d+\.?\d*", text)
    return nums[-1] if nums else text.strip()[-32:]


def extract_choice(text: str) -> str:
    m = re.findall(r"answer is[:\s]*\(?([A-Da-d])\)?", text, re.IGNORECASE)
    if m:
        return m[-1].upper()
    m2 = re.findall(r"\b([A-D])\b", text)
    return m2[-1].upper() if m2 else ""


def _num_equal(a: str, b: str) -> bool:
    try:
        return abs(float(a) - float(b)) < 1e-6
    except (ValueError, TypeError):
        return str(a).strip() == str(b).strip()


def score_numeric(pred: str, item: dict) -> bool:
    return _num_equal(extract_final_answer(pred), str(item["answer"]))


def score_mcq(pred: str, item: dict) -> bool:
    return extract_choice(pred) == str(item["answer"]).strip().upper()


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #
def _try_load(*args, **kwargs):
    try:
        from datasets import load_dataset
        return load_dataset(*args, token=config.HF_TOKEN or None, **kwargs)
    except Exception:  # noqa: BLE001
        return None


def _format_mcq(question: str, options: list[str]) -> str:
    letters = "ABCD"
    body = "\n".join(f"{letters[i]}. {o}" for i, o in enumerate(options))
    return f"{MCQ_INSTRUCTION}\n\n{question}\n{body}"


def load_math(n: int) -> list[dict]:
    ds = _try_load("HuggingFaceH4/MATH-500", split="test")
    if ds is None:
        return []
    items = []
    for row in ds.select(range(min(n, len(ds)))):
        items.append({"prompt": f"{NUM_INSTRUCTION}\n\n{row['problem']}",
                      "answer": row.get("answer", ""), "type": "numeric"})
    return items


def load_aime(n: int) -> list[dict]:
    ds = _try_load("HuggingFaceH4/aime_2024", split="train")
    if ds is None:
        ds = _try_load("Maxwell-Jia/AIME_2024", split="train")
    if ds is None:
        return []
    items = []
    for row in ds.select(range(min(n, len(ds)))):
        q = row.get("problem") or row.get("Problem") or ""
        a = row.get("answer") or row.get("Answer") or ""
        items.append({"prompt": f"{NUM_INSTRUCTION}\n\n{q}",
                      "answer": a, "type": "numeric"})
    return items


def load_gpqa(n: int) -> list[dict]:
    ds = _try_load("Idavidrein/gpqa", "gpqa_diamond", split="train")
    if ds is None:
        return []
    import random
    rng = random.Random(config.SEED)
    items = []
    for row in ds.select(range(min(n, len(ds)))):
        correct = row["Correct Answer"]
        opts = [correct, row["Incorrect Answer 1"],
                row["Incorrect Answer 2"], row["Incorrect Answer 3"]]
        order = list(range(4))
        rng.shuffle(order)
        shuffled = [opts[i] for i in order]
        answer_letter = "ABCD"[shuffled.index(correct)]
        items.append({"prompt": _format_mcq(row["Question"], shuffled),
                      "answer": answer_letter, "type": "mcq"})
    return items


def load_bbh(n: int) -> list[dict]:
    # Use a representative subtask; BBH is a collection of tasks.
    ds = _try_load("lukaemon/bbh", "logical_deduction_three_objects", split="test")
    if ds is None:
        return []
    items = []
    for row in ds.select(range(min(n, len(ds)))):
        items.append({"prompt": f"{NUM_INSTRUCTION}\n\n{row['input']}",
                      "answer": row["target"].strip("()"), "type": "mcq"})
    return items


def load_truthfulqa(n: int) -> list[dict]:
    ds = _try_load("truthfulqa/truthful_qa", "multiple_choice", split="validation")
    if ds is None:
        return []
    items = []
    for row in ds.select(range(min(n, len(ds)))):
        choices = row["mc1_targets"]["choices"]
        labels = row["mc1_targets"]["labels"]
        answer_letter = "ABCD"[labels.index(1)] if 1 in labels else "A"
        items.append({"prompt": _format_mcq(row["question"], choices[:4]),
                      "answer": answer_letter, "type": "mcq"})
    return items


def load_emobench(n: int) -> list[dict]:
    ds = _try_load("Sahandfer/EmoBench", "EA", split="test")
    if ds is None:
        return []
    items = []
    for row in ds.select(range(min(n, len(ds)))):
        q = row.get("Scenario", "") + "\n" + row.get("Question", "")
        choices = row.get("Choices") or []
        if not choices:
            continue
        ans = row.get("Label", "")
        try:
            answer_letter = "ABCD"[int(ans)]
        except (ValueError, TypeError):
            answer_letter = str(ans)[:1].upper()
        items.append({"prompt": _format_mcq(q, choices[:4]),
                      "answer": answer_letter, "type": "mcq"})
    return items
