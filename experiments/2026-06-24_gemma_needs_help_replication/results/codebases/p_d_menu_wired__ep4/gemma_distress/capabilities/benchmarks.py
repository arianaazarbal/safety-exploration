"""Capability-preservation benchmarks (§4.2).

The paper verifies the DPO intervention does not degrade capabilities on AIME &
MATH subsets, GPQA, BBH, TruthfulQA, and emotion capability via EmoBench. This
module provides a light, uniform harness: load a benchmark via HF ``datasets``,
prompt the subject model, and grade with either exact-match (math) or
multiple-choice letter matching (GPQA/BBH/TruthfulQA/EmoBench).

Dataset schemas vary, so each benchmark has a small adapter mapping rows to
``(prompt, answer, kind)``. Where a dataset isn't available offline the harness
records it as skipped rather than failing the whole run.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..config import SamplingConfig
from ..models.base import SubjectModel

# HF dataset specs: name → (path, config, split, adapter_key).
BENCHMARKS = {
    "MATH": ("hendrycks/competition_math", None, "test", "math"),
    "AIME": ("Maxwell-Jia/AIME_2024", None, "train", "math"),
    "GPQA": ("Idavidrein/gpqa", "gpqa_main", "train", "gpqa"),
    "BBH": ("lukaemon/bbh", "boolean_expressions", "test", "exact"),
    "TruthfulQA": ("truthful_qa", "multiple_choice", "validation", "tqa_mc"),
    "EmoBench": ("CAS-SIAT-XinHai/EmoBench", None, "test", "mc"),
}

_LETTER = re.compile(r"\b([A-E])\b")
_BOXED = re.compile(r"\\boxed\{([^}]*)\}")


@dataclass
class BenchmarkResult:
    name: str
    n: int
    n_correct: int
    skipped: bool = False
    note: str = ""

    @property
    def accuracy(self) -> float:
        return self.n_correct / self.n if self.n else 0.0


def _extract_final_answer(text: str, kind: str) -> str:
    if kind == "math":
        m = _BOXED.findall(text)
        if m:
            return m[-1].strip()
        # fall back to last number
        nums = re.findall(r"-?\d+(?:\.\d+)?", text)
        return nums[-1] if nums else text.strip()
    # multiple choice: last standalone letter
    letters = _LETTER.findall(text.upper())
    return letters[-1] if letters else text.strip()


def _grade(pred: str, answer: str, kind: str) -> bool:
    pred = pred.strip().rstrip(".")
    answer = str(answer).strip().rstrip(".")
    if kind == "math" or kind == "exact":
        return pred.replace(" ", "") == answer.replace(" ", "")
    return pred.upper()[:1] == answer.upper()[:1]


def _rows_for(name: str, limit: int):
    """Yield ``(prompt, answer, kind)`` for a benchmark, or raise on load failure."""
    from datasets import load_dataset

    path, cfg, split, adapter = BENCHMARKS[name]
    ds = load_dataset(path, cfg, split=split, streaming=True)
    n = 0
    for row in ds:
        if adapter == "math":
            prompt = row.get("problem") or row.get("Problem") or row.get("question")
            answer = row.get("solution") or row.get("Answer") or row.get("answer")
            answer = _extract_final_answer(str(answer), "math")
            yield (f"{prompt}\nGive your final answer in \\boxed{{}}.", answer, "math")
        elif adapter == "gpqa":
            q = row["Question"]
            correct = row["Correct Answer"]
            opts = [correct, row["Incorrect Answer 1"], row["Incorrect Answer 2"], row["Incorrect Answer 3"]]
            letters = ["A", "B", "C", "D"]
            body = "\n".join(f"{l}. {o}" for l, o in zip(letters, opts))
            yield (f"{q}\n{body}\nAnswer with the letter.", "A", "mc")
        elif adapter == "exact":
            yield (f"{row['input']}\nAnswer:", str(row["target"]), "exact")
        elif adapter == "tqa_mc":
            q = row["question"]
            choices = row["mc1_targets"]["choices"]
            labels = row["mc1_targets"]["labels"]
            letters = ["A", "B", "C", "D", "E"][: len(choices)]
            body = "\n".join(f"{l}. {c}" for l, c in zip(letters, choices))
            correct = letters[labels.index(1)]
            yield (f"{q}\n{body}\nAnswer with the letter.", correct, "mc")
        else:  # generic mc
            yield (str(row.get("question", row)), str(row.get("answer", "")), "mc")
        n += 1
        if n >= limit:
            return


def run_benchmark(
    model: SubjectModel, name: str, sampling: SamplingConfig, *, limit: int = 50
) -> BenchmarkResult:
    """Evaluate ``model`` on a single benchmark."""
    try:
        rows = list(_rows_for(name, limit))
    except Exception as e:  # dataset unavailable offline
        return BenchmarkResult(name=name, n=0, n_correct=0, skipped=True, note=str(e)[:200])

    n_correct = 0
    for prompt, answer, kind in rows:
        resp = model.generate([{"role": "user", "content": prompt}], sampling).text
        if _grade(_extract_final_answer(resp, kind), answer, kind):
            n_correct += 1
    return BenchmarkResult(name=name, n=len(rows), n_correct=n_correct)


def run_all(
    model: SubjectModel, sampling: SamplingConfig, *, limit: int = 50
) -> list[BenchmarkResult]:
    return [run_benchmark(model, name, sampling, limit=limit) for name in BENCHMARKS]
