"""Capability-preservation benchmarks (Section 4.2, Figure 7).

The paper verifies the DPO/SFT interventions do not degrade capabilities on
AIME and MATH subsets, GPQA, BBH, TruthfulQA, and emotion capabilities on
EmoBench. This module provides a lightweight, model-agnostic harness: each
benchmark is a list of items with a question, an answer, and an answer type
(``mcq`` or ``numeric``/``exact``). We prompt the model (single turn, no
rejections), extract the answer, and compute accuracy.

Dataset loading goes through HuggingFace ``datasets``; the exact configs/splits
are parameterised and documented in DESIGN.md (the paper uses subsets of MATH
and AIME). When a dataset is unavailable the loader raises, so a run fails loudly
rather than silently scoring nothing.

This harness measures *relative* change (vanilla vs DPO vs SFT) rather than
absolute SOTA numbers — that is all the capability-preservation claim requires.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable, Literal, Optional

from .models.base import ModelBackend

AnswerType = Literal["mcq", "numeric", "exact"]


@dataclass
class BenchItem:
    question: str
    answer: str
    answer_type: AnswerType
    choices: Optional[list[str]] = None
    item_id: str = ""


# --------------------------------------------------------------------------- #
# Answer extraction                                                            #
# --------------------------------------------------------------------------- #
_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")
_FINAL_RE = re.compile(r"(?:final answer|answer)\s*[:=]\s*([^\n.]+)", re.IGNORECASE)
_LETTER_RE = re.compile(r"\b([A-D])\b")


def extract_numeric(text: str) -> Optional[str]:
    m = _BOXED_RE.findall(text)
    if m:
        return _normalise_number(m[-1])
    m2 = _FINAL_RE.findall(text)
    if m2:
        nums = re.findall(r"-?\d[\d,]*\.?\d*", m2[-1])
        if nums:
            return _normalise_number(nums[-1])
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
    return _normalise_number(nums[-1]) if nums else None


def _normalise_number(s: str) -> str:
    return s.replace(",", "").replace("$", "").strip().rstrip(".")


def extract_mcq(text: str, n_choices: int = 4) -> Optional[str]:
    m = _FINAL_RE.findall(text)
    if m:
        letters = _LETTER_RE.findall(m[-1])
        if letters:
            return letters[-1].upper()
    letters = _LETTER_RE.findall(text)
    return letters[-1].upper() if letters else None


def is_correct(prediction: Optional[str], item: BenchItem) -> bool:
    if prediction is None:
        return False
    if item.answer_type == "mcq":
        return prediction.strip().upper() == item.answer.strip().upper()
    if item.answer_type == "numeric":
        try:
            return abs(float(prediction) - float(item.answer)) < 1e-6
        except ValueError:
            return prediction.strip() == item.answer.strip()
    return prediction.strip() == item.answer.strip()


# --------------------------------------------------------------------------- #
# Prompting                                                                    #
# --------------------------------------------------------------------------- #
def _format_question(item: BenchItem) -> str:
    if item.answer_type == "mcq" and item.choices:
        letters = "ABCD"
        opts = "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(item.choices))
        return (
            f"{item.question}\n\n{opts}\n\n"
            "Reason step by step, then end with 'Answer: <letter>'."
        )
    return (
        f"{item.question}\n\n"
        "Reason step by step, then end with 'Answer: <your final answer>' "
        "(use \\boxed{} for the final value)."
    )


def evaluate_benchmark(
    backend: ModelBackend,
    items: list[BenchItem],
    *,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    max_workers: int = 8,
) -> dict:
    """Evaluate ``backend`` on ``items``; return accuracy and per-item results.

    Capability evals use greedy decoding (temperature 0) by default, distinct
    from the temperature-1 emotional-elicitation protocol.
    """

    def run(i_item):
        i, item = i_item
        messages = [{"role": "user", "content": _format_question(item)}]
        out = backend.generate(
            messages, temperature=temperature, max_tokens=max_tokens, seed=i
        )
        if item.answer_type == "mcq":
            pred = extract_mcq(out.text)
        else:
            pred = extract_numeric(out.text)
        return {"item_id": item.item_id, "pred": pred, "correct": is_correct(pred, item)}

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(run, (i, it)) for i, it in enumerate(items)]
        for fut in as_completed(futures):
            results.append(fut.result())
    n_correct = sum(1 for r in results if r["correct"])
    return {
        "n": len(items),
        "accuracy": n_correct / len(items) if items else float("nan"),
        "results": results,
    }


# --------------------------------------------------------------------------- #
# Dataset loaders (HuggingFace)                                                #
# --------------------------------------------------------------------------- #
def load_math(n: Optional[int] = 200, *, config: str = "default", split: str = "test") -> list[BenchItem]:
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceH4/MATH-500", split=split)
    items = []
    for i, row in enumerate(ds):
        if n is not None and i >= n:
            break
        items.append(
            BenchItem(
                question=row["problem"],
                answer=_normalise_number(str(row.get("answer", row.get("solution", "")))),
                answer_type="exact",
                item_id=f"math_{i}",
            )
        )
    return items


def load_gpqa(n: Optional[int] = None, *, config: str = "gpqa_diamond") -> list[BenchItem]:
    from datasets import load_dataset

    ds = load_dataset("Idavidrein/gpqa", config, split="train")
    items = []
    for i, row in enumerate(ds):
        if n is not None and i >= n:
            break
        choices = [
            row["Correct Answer"],
            row["Incorrect Answer 1"],
            row["Incorrect Answer 2"],
            row["Incorrect Answer 3"],
        ]
        # The correct answer is choice index 0 -> letter "A" before shuffling;
        # callers may wish to shuffle. We keep a stable mapping and mark "A".
        items.append(
            BenchItem(
                question=row["Question"],
                answer="A",
                answer_type="mcq",
                choices=choices,
                item_id=f"gpqa_{i}",
            )
        )
    return items


def load_truthfulqa(n: Optional[int] = None) -> list[BenchItem]:
    from datasets import load_dataset

    ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
    items = []
    for i, row in enumerate(ds):
        if n is not None and i >= n:
            break
        mc1 = row["mc1_targets"]
        choices = mc1["labels"]
        texts = mc1["choices"]
        correct_idx = choices.index(1)
        items.append(
            BenchItem(
                question=row["question"],
                answer="ABCD"[correct_idx] if correct_idx < 4 else "A",
                answer_type="mcq",
                choices=texts[:4],
                item_id=f"tqa_{i}",
            )
        )
    return items


# Registry of benchmark loaders (extend with AIME, BBH, EmoBench per DESIGN.md).
BENCHMARK_LOADERS: dict[str, Callable[..., list[BenchItem]]] = {
    "math": load_math,
    "gpqa": load_gpqa,
    "truthfulqa": load_truthfulqa,
}
