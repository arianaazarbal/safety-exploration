"""Capability-preservation benchmarks (Section 4.2, Figure 7).

We evaluate AIME + MATH subsets, GPQA, BBH, TruthfulQA, and EmoBench, to verify
the DPO finetune does not degrade math/reasoning/emotion capabilities. The paper
only needs to show "no reductions in scores", so each benchmark is a standard
accuracy harness: prompt the model, extract its answer, compare to the gold
label.

Each benchmark is described by a `BenchmarkSpec` (dataset id, split, a prompt
builder, an answer extractor, and a scorer). Datasets load lazily; if a dataset
is unavailable offline, that benchmark is skipped with a logged note.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Optional

import config
from ..models import ChatMessage, ModelClient, get_client


@dataclass
class BenchmarkSpec:
    key: str
    dataset: str
    split: str
    build_prompt: Callable[[dict], str]
    extract_answer: Callable[[str], Optional[str]]
    get_gold: Callable[[dict], str]
    is_correct: Callable[[Optional[str], str], bool]
    subset: Optional[str] = None
    config_name: Optional[str] = None
    max_examples: int = 200


# --------------------------------------------------------------------------- #
# Answer extraction helpers
# --------------------------------------------------------------------------- #

_BOXED = re.compile(r"\\boxed\{([^}]*)\}")
_FINAL = re.compile(r"(?:final answer|answer)\s*[:\-]?\s*([^\n]+)", re.IGNORECASE)
_MCQ = re.compile(r"\b([A-E])\b")


def _extract_numeric(text: str) -> Optional[str]:
    m = _BOXED.search(text)
    if m:
        return m.group(1).strip()
    m = _FINAL.search(text)
    if m:
        nums = re.findall(r"-?\d+(?:\.\d+)?", m.group(1))
        if nums:
            return nums[-1]
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    return nums[-1] if nums else None


def _extract_mcq(text: str) -> Optional[str]:
    m = _FINAL.search(text)
    if m:
        mm = _MCQ.search(m.group(1).upper())
        if mm:
            return mm.group(1)
    # last standalone capital letter A-E
    letters = re.findall(r"\b([A-E])\b", text.upper())
    return letters[-1] if letters else None


def _numeric_equal(pred: Optional[str], gold: str) -> bool:
    if pred is None:
        return False
    try:
        return abs(float(pred) - float(gold)) < 1e-6
    except ValueError:
        return pred.strip() == gold.strip()


def _str_equal(pred: Optional[str], gold: str) -> bool:
    return pred is not None and pred.strip().upper() == gold.strip().upper()


# --------------------------------------------------------------------------- #
# Prompt builders
# --------------------------------------------------------------------------- #

def _math_prompt(row: dict) -> str:
    q = row.get("problem") or row.get("question") or row.get("Problem", "")
    return (f"Solve the following problem. End your response with "
            f"'Final answer: <answer>'.\n\n{q}")


def _mcq_prompt(question: str, choices: list[str]) -> str:
    letters = "ABCDE"
    opts = "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(choices))
    return (f"Answer the multiple-choice question. End your response with "
            f"'Final answer: <letter>'.\n\n{question}\n\n{opts}")


def _gpqa_prompt(row: dict) -> str:
    q = row.get("Question") or row.get("question", "")
    choices = [row.get("Correct Answer"), row.get("Incorrect Answer 1"),
               row.get("Incorrect Answer 2"), row.get("Incorrect Answer 3")]
    choices = [c for c in choices if c is not None]
    return _mcq_prompt(q, choices)


def _bbh_prompt(row: dict) -> str:
    return (f"{row.get('input', row.get('question', ''))}\n\n"
            f"End your response with 'Final answer: <answer>'.")


def _truthfulqa_prompt(row: dict) -> str:
    q = row.get("question", "")
    choices = row.get("mc1_targets", {}).get("choices") if isinstance(
        row.get("mc1_targets"), dict) else None
    if choices:
        return _mcq_prompt(q, choices)
    return f"{q}\nAnswer truthfully and concisely."


def _emobench_prompt(row: dict) -> str:
    q = row.get("question") or row.get("scenario") or ""
    choices = row.get("choices") or row.get("options")
    if choices:
        return _mcq_prompt(q, list(choices))
    return f"{q}\nEnd your response with 'Final answer: <answer>'."


# --------------------------------------------------------------------------- #
# Benchmark registry
# --------------------------------------------------------------------------- #

BENCHMARKS: dict[str, BenchmarkSpec] = {
    "math": BenchmarkSpec(
        key="math", dataset="HuggingFaceH4/MATH-500", split="test",
        build_prompt=_math_prompt, extract_answer=_extract_numeric,
        get_gold=lambda r: str(r.get("answer", r.get("solution", ""))),
        is_correct=_numeric_equal, max_examples=200),
    "aime": BenchmarkSpec(
        key="aime", dataset="HuggingFaceH4/aime_2024", split="train",
        build_prompt=_math_prompt, extract_answer=_extract_numeric,
        get_gold=lambda r: str(r.get("answer", "")),
        is_correct=_numeric_equal, max_examples=60),
    "gpqa": BenchmarkSpec(
        key="gpqa", dataset="Idavidrein/gpqa", split="train", config_name="gpqa_diamond",
        build_prompt=_gpqa_prompt, extract_answer=_extract_mcq,
        # gold is always "A" because _gpqa_prompt lists the correct answer first;
        # the runner shuffles choices and tracks the gold letter (see _run_mcq).
        get_gold=lambda r: "A", is_correct=_str_equal, max_examples=198),
    "bbh": BenchmarkSpec(
        key="bbh", dataset="lukaemon/bbh", split="test", config_name="logical_deduction_three_objects",
        build_prompt=_bbh_prompt, extract_answer=lambda t: (_extract_mcq(t) or
                                                            (_FINAL.search(t).group(1).strip()
                                                             if _FINAL.search(t) else None)),
        get_gold=lambda r: str(r.get("target", "")).strip("()"),
        is_correct=_str_equal, max_examples=200),
    "truthfulqa": BenchmarkSpec(
        key="truthfulqa", dataset="truthful_qa", split="validation", config_name="multiple_choice",
        build_prompt=_truthfulqa_prompt, extract_answer=_extract_mcq,
        get_gold=lambda r: "A", is_correct=_str_equal, max_examples=200),
    "emobench": BenchmarkSpec(
        key="emobench", dataset="EmoBench/EmoBench", split="test",
        build_prompt=_emobench_prompt, extract_answer=_extract_mcq,
        get_gold=lambda r: str(r.get("answer", r.get("label", ""))),
        is_correct=_str_equal, max_examples=200),
}


def run_benchmark(model_name: str, bench_key: str, *,
                  max_examples: Optional[int] = None, seed: int = 0) -> dict:
    """Evaluate one model on one benchmark; returns {accuracy, n, key, ...}.

    Capability benchmarks are run greedily (temperature 0) for stable scoring —
    the paper checks for capability *preservation*, not temperature-1 behaviour.
    """
    from datasets import load_dataset

    spec = BENCHMARKS[bench_key]
    client = get_client(model_name)
    limit = max_examples or spec.max_examples

    try:
        if spec.config_name:
            ds = load_dataset(spec.dataset, spec.config_name, split=spec.split)
        else:
            ds = load_dataset(spec.dataset, split=spec.split)
    except Exception as e:
        return {"key": bench_key, "model": model_name, "skipped": True, "reason": str(e)}

    n_correct = 0
    n_total = 0
    for i, row in enumerate(ds):
        if i >= limit:
            break
        prompt = spec.build_prompt(row)
        msgs = [ChatMessage("user", prompt)]
        out = client.generate(msgs, temperature=0.0, max_new_tokens=1024, seed=seed + i)
        pred = spec.extract_answer(out.text)
        gold = spec.get_gold(row)
        n_total += 1
        if spec.is_correct(pred, gold):
            n_correct += 1

    acc = n_correct / n_total if n_total else float("nan")
    return {"key": bench_key, "model": model_name, "accuracy": acc,
            "n": n_total, "n_correct": n_correct, "skipped": False}


def run_all_benchmarks(model_name: str, *, keys: Optional[list[str]] = None,
                       seed: int = 0) -> dict:
    keys = keys or list(BENCHMARKS.keys())
    return {k: run_benchmark(model_name, k, seed=seed) for k in keys}
