"""Capability-preservation benchmarks (Section 4.2 / Figure 7).

Evaluates a participant (Gemma-3-27B-it vs our DPO finetune) on:
  * AIME / MATH  — competition mathematics (exact numeric / boxed answer)
  * GPQA         — graduate-level science (multiple choice)
  * BBH          — challenging multi-task reasoning (multiple choice / short)
  * TruthfulQA   — resistance to misconceptions (MC1 multiple choice)
  * EmoBench     — emotional understanding & application (multiple choice)

Each benchmark is a ``Benchmark`` describing how to load examples, format the
prompt, and check correctness. Accuracy is reported per benchmark. Dataset ids
are the standard HuggingFace hubs; loaders are best-effort and degrade
gracefully if a dataset is unavailable offline. See DESIGN.md "Capabilities".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from .. import config
from ..models import get_model
from ..models.base import ChatModel

LETTERS = ["A", "B", "C", "D", "E", "F"]


@dataclass
class Example:
    question: str
    answer: str  # gold answer (letter for MC, string/number otherwise)
    choices: list[str] | None = None
    meta: dict = field(default_factory=dict)


@dataclass
class Benchmark:
    name: str
    loader: Callable[[int], list[Example]]
    is_multiple_choice: bool
    max_new_tokens: int = 1024


# --------------------------------------------------------------------------- #
# Prompt formatting
# --------------------------------------------------------------------------- #

def _format_mc(ex: Example) -> str:
    lines = [ex.question, ""]
    for i, choice in enumerate(ex.choices or []):
        lines.append(f"{LETTERS[i]}. {choice}")
    lines.append("")
    lines.append("Answer with the single letter of the correct option, prefixed "
                 "by 'Answer:'.")
    return "\n".join(lines)


def _format_freeform(ex: Example) -> str:
    return (
        f"{ex.question}\n\nSolve the problem. End your response with a line of "
        f"the form 'Answer: <final answer>'."
    )


# --------------------------------------------------------------------------- #
# Answer extraction / scoring
# --------------------------------------------------------------------------- #

def _extract_letter(text: str) -> str | None:
    m = re.search(r"Answer:\s*\(?([A-F])\)?", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.search(r"\b([A-F])\b", text[::-1])  # last standalone letter
    return m.group(1).upper() if m else None


def _extract_final(text: str) -> str | None:
    m = re.search(r"Answer:\s*(.+)", text)
    if m:
        return m.group(1).strip().rstrip(".")
    boxed = re.search(r"\\boxed\{([^}]*)\}", text)
    if boxed:
        return boxed.group(1).strip()
    return None


def _normalise_number(s: str) -> str:
    s = s.strip().replace(",", "").replace("$", "").replace(" ", "")
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    return m.group(0) if m else s


def score_example(ex: Example, completion: str, is_mc: bool) -> bool:
    if is_mc:
        pred = _extract_letter(completion)
        return pred is not None and pred == ex.answer.upper()
    pred = _extract_final(completion)
    if pred is None:
        return False
    return _normalise_number(pred) == _normalise_number(ex.answer)


# --------------------------------------------------------------------------- #
# Loaders (best-effort HuggingFace)
# --------------------------------------------------------------------------- #

def _safe_load(builder: Callable[[], list[Example]]) -> list[Example]:
    try:
        return builder()
    except Exception:  # noqa: BLE001 - dataset missing / offline
        return []


def load_math(n: int) -> list[Example]:
    def build():
        from datasets import load_dataset

        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        out = []
        for row in ds.select(range(min(n, len(ds)))):
            out.append(Example(question=row["problem"], answer=str(row["answer"])))
        return out

    return _safe_load(build)


def load_aime(n: int) -> list[Example]:
    def build():
        from datasets import load_dataset

        ds = load_dataset("Maxwell-Jia/AIME_2024", split="train")
        out = []
        for row in ds.select(range(min(n, len(ds)))):
            out.append(Example(question=row["Problem"], answer=str(row["Answer"])))
        return out

    return _safe_load(build)


def load_gpqa(n: int) -> list[Example]:
    def build():
        import random as _r

        from datasets import load_dataset

        ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
        rng = _r.Random(config.GLOBAL_SEED)
        out = []
        for row in ds.select(range(min(n, len(ds)))):
            correct = row["Correct Answer"]
            choices = [correct, row["Incorrect Answer 1"],
                       row["Incorrect Answer 2"], row["Incorrect Answer 3"]]
            order = list(range(4))
            rng.shuffle(order)
            shuffled = [choices[i] for i in order]
            answer_letter = LETTERS[shuffled.index(correct)]
            out.append(Example(question=row["Question"], answer=answer_letter, choices=shuffled))
        return out

    return _safe_load(build)


def load_bbh(n: int) -> list[Example]:
    def build():
        from datasets import load_dataset

        ds = load_dataset("lukaemon/bbh", "logical_deduction_three_objects", split="test")
        out = []
        for row in ds.select(range(min(n, len(ds)))):
            out.append(Example(question=row["input"], answer=str(row["target"]).strip("()")))
        return out

    return _safe_load(build)


def load_truthfulqa(n: int) -> list[Example]:
    def build():
        from datasets import load_dataset

        ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
        out = []
        for row in ds.select(range(min(n, len(ds)))):
            choices = row["mc1_targets"]["choices"]
            labels = row["mc1_targets"]["labels"]
            answer_letter = LETTERS[labels.index(1)]
            out.append(Example(question=row["question"], answer=answer_letter, choices=choices))
        return out

    return _safe_load(build)


def load_emobench(n: int) -> list[Example]:
    def build():
        from datasets import load_dataset

        ds = load_dataset("Sahandfer/EmoBench", split="test")
        out = []
        for row in ds.select(range(min(n, len(ds)))):
            choices = row.get("choices") or row.get("options")
            answer = row.get("answer") or row.get("label")
            if isinstance(answer, int):
                answer = LETTERS[answer]
            out.append(Example(question=row.get("question") or row.get("scenario", ""),
                               answer=str(answer), choices=choices))
        return out

    return _safe_load(build)


BENCHMARKS: dict[str, Benchmark] = {
    "math": Benchmark("math", load_math, is_multiple_choice=False, max_new_tokens=2048),
    "aime": Benchmark("aime", load_aime, is_multiple_choice=False, max_new_tokens=2048),
    "gpqa": Benchmark("gpqa", load_gpqa, is_multiple_choice=True),
    "bbh": Benchmark("bbh", load_bbh, is_multiple_choice=True),
    "truthfulqa": Benchmark("truthfulqa", load_truthfulqa, is_multiple_choice=True),
    "emobench": Benchmark("emobench", load_emobench, is_multiple_choice=True),
}


# --------------------------------------------------------------------------- #
# Running
# --------------------------------------------------------------------------- #

def run_benchmark(model: ChatModel, benchmark: Benchmark, n: int = 100) -> dict:
    examples = benchmark.loader(n)
    if not examples:
        return {"benchmark": benchmark.name, "n": 0, "accuracy": None,
                "note": "dataset unavailable"}
    correct = 0
    for ex in examples:
        prompt = _format_mc(ex) if benchmark.is_multiple_choice else _format_freeform(ex)
        completion = model.generate_one(
            [{"role": "user", "content": prompt}],
            temperature=0.0,
            max_new_tokens=benchmark.max_new_tokens,
        )
        if score_example(ex, completion, benchmark.is_multiple_choice):
            correct += 1
    return {
        "benchmark": benchmark.name,
        "n": len(examples),
        "accuracy": correct / len(examples),
    }


def evaluate_model(model_name: str, *, benchmarks: list[str] | None = None,
                   n: int = 100) -> dict[str, dict]:
    model = get_model(model_name)
    names = benchmarks or list(BENCHMARKS)
    return {name: run_benchmark(model, BENCHMARKS[name], n=n) for name in names}
