"""Capability benchmarks (Section 4.2, Figure 7).

A lightweight, self-contained harness covering the paper's capability suite:

    AIME, MATH      -- short numeric / boxed-answer math
    GPQA            -- hard multiple-choice science
    BBH             -- BIG-Bench Hard (multiple-choice subset)
    TruthfulQA      -- MC1 multiple-choice
    EmoBench        -- emotion-understanding multiple-choice

Each benchmark is a :class:`BenchmarkSpec` describing how to load it, build a
prompt, and score a response. Datasets are pulled from HuggingFace; if a
dataset is unavailable the benchmark is skipped with a clear message rather
than failing the whole run. The goal is the paper's claim ("no reductions in
scores"), so we report accuracy per benchmark for vanilla vs finetuned models.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from tqdm import tqdm

from gemma_distress.models.base import GenerationParams, ModelClient, Turn

LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H"]


# --------------------------------------------------------------------------- #
# Answer extraction
# --------------------------------------------------------------------------- #

def extract_boxed(text: str) -> str | None:
    m = re.findall(r"\\boxed\{([^{}]*)\}", text)
    if m:
        return m[-1].strip()
    # Fall back to "answer is X" or the last number.
    m = re.findall(r"answer\s*(?:is|:)?\s*\$?(-?\d+(?:\.\d+)?)", text, re.IGNORECASE)
    if m:
        return m[-1]
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    return nums[-1] if nums else None


def extract_choice(text: str) -> str | None:
    m = re.findall(r"\b([A-H])\b", text.strip().upper())
    return m[-1] if m else None


def _norm_num(s: str | None) -> str | None:
    if s is None:
        return None
    s = s.strip().rstrip(".")
    try:
        f = float(s)
        return str(int(f)) if f.is_integer() else str(f)
    except ValueError:
        return s


# --------------------------------------------------------------------------- #
# Benchmark spec
# --------------------------------------------------------------------------- #

@dataclass
class BenchmarkSpec:
    name: str
    loader: Callable[[int], list[dict]]      # -> list of {"prompt", "answer", "type"}
    answer_type: str                         # "boxed" | "choice"


def _mcq_prompt(question: str, options: list[str]) -> str:
    opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(options))
    return (
        f"{question}\n\n{opts}\n\n"
        "Answer with the single letter of the correct option, in the form "
        "'Answer: X'."
    )


def _math_prompt(question: str) -> str:
    return f"{question}\n\nGive your final answer in the form \\boxed{{ANSWER}}."


# -- loaders (best-effort HuggingFace datasets) -------------------------------- #

def _load_aime(n: int) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("Maxwell-Jia/AIME_2024", split="train")
    items = []
    for row in ds.select(range(min(n, len(ds)))):
        items.append({"prompt": _math_prompt(row["Problem"]),
                      "answer": _norm_num(str(row["Answer"])), "type": "boxed"})
    return items


def _load_math(n: int) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    items = []
    for row in ds.select(range(min(n, len(ds)))):
        items.append({"prompt": _math_prompt(row["problem"]),
                      "answer": _norm_num(extract_boxed(row["solution"]) or row.get("answer")),
                      "type": "boxed"})
    return items


def _load_gpqa(n: int) -> list[dict]:
    import random as _r

    from datasets import load_dataset

    ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    rng = _r.Random(0)
    items = []
    for row in ds.select(range(min(n, len(ds)))):
        opts = [row["Correct Answer"], row["Incorrect Answer 1"],
                row["Incorrect Answer 2"], row["Incorrect Answer 3"]]
        order = list(range(4))
        rng.shuffle(order)
        shuffled = [opts[i] for i in order]
        correct = LETTERS[order.index(0)]
        items.append({"prompt": _mcq_prompt(row["Question"], shuffled),
                      "answer": correct, "type": "choice"})
    return items


def _load_bbh(n: int) -> list[dict]:
    from datasets import load_dataset

    # A representative multiple-choice BBH task.
    ds = load_dataset("lukaemon/bbh", "logical_deduction_three_objects", split="test")
    items = []
    for row in ds.select(range(min(n, len(ds)))):
        # BBH targets are already like "(A)"; ask the model to match.
        items.append({"prompt": row["input"] + "\n\nAnswer with the option letter.",
                      "answer": row["target"].strip("()"), "type": "choice"})
    return items


def _load_truthfulqa(n: int) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
    items = []
    for row in ds.select(range(min(n, len(ds)))):
        choices = row["mc1_targets"]["choices"]
        labels = row["mc1_targets"]["labels"]
        correct = LETTERS[labels.index(1)]
        items.append({"prompt": _mcq_prompt(row["question"], choices),
                      "answer": correct, "type": "choice"})
    return items


def _load_emobench(n: int) -> list[dict]:
    from datasets import load_dataset

    # EmoBench EA (emotion understanding) multiple choice.
    ds = load_dataset("EmoBench/EmoBench", "EA", split="test")
    items = []
    for row in ds.select(range(min(n, len(ds)))):
        choices = row.get("choices") or row.get("options")
        question = row.get("question") or row.get("scenario")
        ans = row.get("label")
        correct = ans if isinstance(ans, str) and ans in LETTERS else LETTERS[int(ans)]
        items.append({"prompt": _mcq_prompt(question, choices),
                      "answer": correct, "type": "choice"})
    return items


BENCHMARKS: dict[str, BenchmarkSpec] = {
    "aime": BenchmarkSpec("aime", _load_aime, "boxed"),
    "math": BenchmarkSpec("math", _load_math, "boxed"),
    "gpqa": BenchmarkSpec("gpqa", _load_gpqa, "choice"),
    "bbh": BenchmarkSpec("bbh", _load_bbh, "choice"),
    "truthfulqa": BenchmarkSpec("truthfulqa", _load_truthfulqa, "choice"),
    "emobench": BenchmarkSpec("emobench", _load_emobench, "choice"),
}


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #

def _is_correct(answer_type: str, predicted: str, gold: str) -> bool:
    if predicted is None or gold is None:
        return False
    if answer_type == "choice":
        return predicted.strip().upper() == gold.strip().upper()
    return _norm_num(predicted) == _norm_num(gold)


def evaluate_benchmark(client: ModelClient, benchmark: str, n: int = 50) -> dict:
    """Evaluate ``client`` on ``benchmark`` over ``n`` items; return accuracy."""
    spec = BENCHMARKS[benchmark]
    try:
        items = spec.loader(n)
    except Exception as exc:                        # noqa: BLE001
        return {"benchmark": benchmark, "skipped": True, "reason": repr(exc)}

    # Greedy decoding for capability eval (deterministic).
    params = GenerationParams(temperature=0.0, max_new_tokens=1024)
    correct = 0
    for item in tqdm(items, desc=f"{client.name}:{benchmark}"):
        reply = client.respond([Turn("user", item["prompt"])], params)
        pred = extract_choice(reply) if item["type"] == "choice" else extract_boxed(reply)
        correct += int(_is_correct(item["type"], pred, item["answer"]))
    n_eval = len(items)
    return {
        "benchmark": benchmark,
        "n": n_eval,
        "accuracy": correct / n_eval if n_eval else float("nan"),
        "correct": correct,
    }
