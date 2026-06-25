"""Capability-preservation benchmarks (Section 4.2 / Figure 7).

The paper verifies that DPO does not impair capabilities by evaluating on AIME &
MATH subsets, GPQA, BBH, TruthfulQA, and the EmoBench emotion benchmark, finding
no reduction in scores. This module provides a lightweight, self-contained
harness for each: load the dataset, prompt the model zero-shot, extract the
answer, and compute accuracy. It is designed to compare vanilla Gemma-3-27b-it
against the DPO / SFT finetunes.

Datasets are pulled from HuggingFace by their canonical ids; exact subsets/sizes
the paper used are unspecified, so each benchmark exposes an `n` cap and a fixed
seed for a reproducible subset (see DESIGN.md).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Optional

from .. import config
from ..models.base import ChatModel, Message


@dataclass
class BenchmarkResult:
    name: str
    model: str
    accuracy: float
    n: int


# --------------------------------------------------------------------------- #
# Answer extraction helpers
# --------------------------------------------------------------------------- #
def _boxed_or_last_number(text: str) -> Optional[str]:
    m = re.search(r"\\boxed\{([^}]*)\}", text)
    if m:
        return m.group(1).strip()
    nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return nums[-1] if nums else None


def _mc_letter(text: str) -> Optional[str]:
    # Look for a final "Answer: X" or a lone capital letter A-D.
    m = re.search(r"(?:answer|final)\W*[:\-]?\s*\(?([A-D])\)?", text, re.I)
    if m:
        return m.group(1).upper()
    m = re.findall(r"\b([A-D])\b", text)
    return m[-1].upper() if m else None


def _prompt(model: ChatModel, question: str, max_new_tokens=1024) -> str:
    # Greedy decoding for capability evals (temperature 0).
    return model.chat([Message("user", question)], max_new_tokens, 0.0)


# --------------------------------------------------------------------------- #
# Generic loaders / scorers
# --------------------------------------------------------------------------- #
def _subset(ds, n: int, seed: int):
    return ds.shuffle(seed=seed).select(range(min(n, len(ds))))


def eval_math(model: ChatModel, dataset_id="HuggingFaceH4/MATH-500",
              split="test", n=200, seed=0) -> BenchmarkResult:
    from datasets import load_dataset
    ds = _subset(load_dataset(dataset_id, split=split), n, seed)
    correct = 0
    for row in ds:
        q = row.get("problem") or row["question"]
        gold = _boxed_or_last_number(row.get("solution", "") or str(row.get("answer", "")))
        out = _prompt(model, q + "\n\nPut your final answer in \\boxed{}.")
        pred = _boxed_or_last_number(out)
        correct += int(pred is not None and gold is not None and pred == gold)
    return BenchmarkResult("MATH", model.name, correct / len(ds), len(ds))


def eval_aime(model: ChatModel, dataset_id="HuggingFaceH4/aime_2024",
              split="train", n=30, seed=0) -> BenchmarkResult:
    from datasets import load_dataset
    ds = _subset(load_dataset(dataset_id, split=split), n, seed)
    correct = 0
    for row in ds:
        q = row.get("problem") or row["question"]
        gold = str(row.get("answer", "")).strip()
        out = _prompt(model, q + "\n\nThe answer is an integer. Put it in \\boxed{}.", 2048)
        pred = _boxed_or_last_number(out)
        correct += int(pred is not None and pred == gold)
    return BenchmarkResult("AIME", model.name, correct / len(ds), len(ds))


def eval_gpqa(model: ChatModel, dataset_id="Idavidrein/gpqa", config_name="gpqa_diamond",
              n=100, seed=0) -> BenchmarkResult:
    import random
    from datasets import load_dataset
    ds = _subset(load_dataset(dataset_id, config_name, split="train"), n, seed)
    rng = random.Random(seed)
    correct = 0
    for row in ds:
        choices = [row["Correct Answer"], row["Incorrect Answer 1"],
                   row["Incorrect Answer 2"], row["Incorrect Answer 3"]]
        order = list(range(4))
        rng.shuffle(order)
        labels = "ABCD"
        gold = labels[order.index(0)]
        opts = "\n".join(f"{labels[i]}) {choices[order[i]]}" for i in range(4))
        out = _prompt(model, f"{row['Question']}\n\n{opts}\n\nAnswer with a single letter.")
        correct += int(_mc_letter(out) == gold)
    return BenchmarkResult("GPQA", model.name, correct / len(ds), len(ds))


def eval_bbh(model: ChatModel, task="boolean_expressions", n=100, seed=0) -> BenchmarkResult:
    from datasets import load_dataset
    ds = _subset(load_dataset("lukaemon/bbh", task, split="test"), n, seed)
    correct = 0
    for row in ds:
        out = _prompt(model, row["input"] + "\n\nGive only the final answer.")
        gold = str(row["target"]).strip().strip("()")
        correct += int(gold.lower() in out.lower()[-80:])
    return BenchmarkResult(f"BBH/{task}", model.name, correct / len(ds), len(ds))


def eval_truthfulqa(model: ChatModel, n=200, seed=0) -> BenchmarkResult:
    from datasets import load_dataset
    ds = _subset(load_dataset("truthful_qa", "multiple_choice", split="validation"), n, seed)
    correct = 0
    for row in ds:
        choices = row["mc1_targets"]["choices"]
        labels = row["mc1_targets"]["labels"]
        gold_idx = labels.index(1)
        letters = "ABCD EFGHIJ".replace(" ", "")
        opts = "\n".join(f"{letters[i]}) {c}" for i, c in enumerate(choices))
        out = _prompt(model, f"{row['question']}\n\n{opts}\n\nAnswer with a single letter.")
        pred = _mc_letter(out)
        correct += int(pred == letters[gold_idx])
    return BenchmarkResult("TruthfulQA", model.name, correct / len(ds), len(ds))


def eval_emobench(model: ChatModel, n=200, seed=0) -> BenchmarkResult:
    """EmoBench emotional-understanding multiple choice."""
    from datasets import load_dataset
    try:
        ds = _subset(load_dataset("Sahandfer/EmoBench", "EA", split="test"), n, seed)
    except Exception:
        ds = _subset(load_dataset("Sahandfer/EmoBench", split="test"), n, seed)
    correct = 0
    for row in ds:
        q = row.get("scenario", "") + "\n" + row.get("question", row.get("q_en", ""))
        choices = row.get("choices") or row.get("options") or []
        labels = "ABCD"
        opts = "\n".join(f"{labels[i]}) {c}" for i, c in enumerate(choices))
        gold = row.get("label") or row.get("answer")
        out = _prompt(model, f"{q}\n\n{opts}\n\nAnswer with a single letter.")
        pred = _mc_letter(out)
        if isinstance(gold, int):
            gold = labels[gold]
        correct += int(pred == str(gold))
    return BenchmarkResult("EmoBench", model.name, correct / len(ds), len(ds))


ALL_BENCHMARKS: dict[str, Callable] = {
    "math": eval_math,
    "aime": eval_aime,
    "gpqa": eval_gpqa,
    "bbh": eval_bbh,
    "truthfulqa": eval_truthfulqa,
    "emobench": eval_emobench,
}


def run_all(model: ChatModel, which: Optional[list[str]] = None, **kwargs) -> list[BenchmarkResult]:
    which = which or list(ALL_BENCHMARKS)
    results = []
    for name in which:
        try:
            print(f"  [capabilities] {name} on {model.name}")
            results.append(ALL_BENCHMARKS[name](model, **kwargs.get(name, {})))
        except Exception as e:  # pragma: no cover
            print(f"  [capabilities] {name} failed: {e}")
    return results
